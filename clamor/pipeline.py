"""End-to-end orchestration.

The work is split in two stages so interactive tools stay fast:

``build_theme_model``  the expensive part (embedding + clustering). Depends only on text.
``analyze``            joins revenue, detects trends, evaluates releases and scores themes.
                       Cheap enough to rerun whenever the user moves a slider.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import Config, Weights
from .embeddings import Embedder, get_embedder
from .impact import match_releases, release_radar
from .io import load_accounts, load_feedback, load_releases
from .lang import LanguagePack, get_language, with_ascii_variants
from .privacy import redact
from .scoring import score_themes
from .sentiment import score_feedback
from .text import boilerplate_mask, segment_feedback
from .themes import OTHER, ThemeModel, cluster_vectors, consolidate, describe_themes
from .trends import detect_trends, weekly_mentions

NEGATIVE = -0.2


@dataclass
class Analysis:
    feedback: pd.DataFrame  # input + sentiment, primary theme, plan, mrr
    mentions: pd.DataFrame  # one row per (doc, theme)
    themes: pd.DataFrame  # the ranked roadmap
    weekly: pd.DataFrame
    releases: pd.DataFrame | None
    side_effects: pd.DataFrame | None
    model: ThemeModel
    config: Config
    as_of: pd.Timestamp
    has_revenue: bool
    inputs: dict  # validated input tables, to re-run cheaply with another model or date

    @property
    def roadmap(self) -> pd.DataFrame:
        return self.themes[self.themes["score"].notna()]

    @property
    def people(self) -> str:
        """What the IDs stand for: paying accounts, or the users of a consumer app."""
        return "accounts" if self.has_revenue else "users"


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)


def build_theme_model(
    feedback: pd.DataFrame,
    config: Config = Config(),
    embedder: Embedder | None = None,
    sentiment: pd.Series | None = None,
) -> ThemeModel:
    feedback = feedback.reset_index(drop=True)
    lang = get_language(config.language)
    embedder = embedder or get_embedder(config.backend, lang=lang)
    backend = embedder.name
    if config.segment_sentences:
        segments = segment_feedback(feedback, config.product_names, lang)
    else:  # ablation: one "segment" per item, the way most feedback tools work
        segments = pd.DataFrame(
            {"doc": range(len(feedback)), "position": 0, "text": feedback["text"].astype(str)}
        )
    texts = segments["text"].tolist()
    protos = list(lang.boilerplate_prototypes)
    if lang.code == "tr":  # informal reviews often skip Turkish characters
        protos = list(with_ascii_variants(lang.boilerplate_prototypes))
    embedder.fit(texts + protos)
    vectors = embedder.encode(texts)
    prototypes = embedder.encode(protos)

    b_thr = config.boilerplate_threshold or config.backend_default("boilerplate_threshold", backend)
    boiler = boilerplate_mask(vectors, prototypes, b_thr)
    # an item made only of boilerplate ("Great app, thanks!") keeps its segments as content
    all_boiler = pd.Series(boiler).groupby(segments["doc"]).transform("all").to_numpy()
    content = ~boiler | all_boiler

    min_size = max(5, int(round(config.min_theme_share * len(feedback))))
    d_thr = config.distance_threshold or config.backend_default("distance_threshold", backend)
    labels = np.full(len(segments), OTHER)
    labels[content] = cluster_vectors(
        vectors[content],
        distance_threshold=d_thr,
        min_size=min_size,
        reassign_similarity=config.reassign_similarity,
        random_state=config.random_state,
    )
    if config.consolidate:
        semantic = getattr(embedder, "semantic_part", lambda v: v)(vectors)
        labels[content] = consolidate(
            labels[content],
            [texts[i] for i in np.flatnonzero(content)],
            semantic[content],
            lang=lang,
        )
    segments["is_boilerplate"] = ~content
    segments["cluster"] = labels

    if sentiment is None:
        sentiment = score_feedback(feedback, lang=lang)
    return _finalize(
        segments, vectors, sentiment.to_numpy(), len(feedback), backend, embedder, lang
    )


def _finalize(
    segments: pd.DataFrame,
    vectors: np.ndarray,
    doc_sentiment: np.ndarray,
    n_docs: int,
    backend: str,
    embedder: Embedder | None,
    lang: LanguagePack,
) -> ThemeModel:
    """Describe the clusters in `segments["cluster"]` and derive per-item theme mentions."""
    segments = segments.copy()
    content = ~segments["is_boilerplate"].to_numpy()
    labels = segments["cluster"].to_numpy()
    themes, centroids = describe_themes(
        segments[content].reset_index(drop=True),
        vectors[content],
        labels[content],
        doc_sentiment,
        lang=lang,
    )
    id_of = dict(zip(themes["cluster"], themes["theme_id"], strict=True))
    segments["theme_id"] = segments["cluster"].map(id_of)

    tagged = segments[segments["theme_id"].notna()]
    doc_themes = (
        tagged.sort_values(["doc", "position"])
        .drop_duplicates(["doc", "theme_id"])
        .loc[:, ["doc", "theme_id"]]
        .reset_index(drop=True)
    )
    doc_themes["primary"] = ~doc_themes["doc"].duplicated()
    primary = np.full(n_docs, None, dtype=object)
    first = doc_themes[doc_themes["primary"]]
    primary[first["doc"].to_numpy()] = first["theme_id"].to_numpy()

    return ThemeModel(
        segments=segments,
        doc_themes=doc_themes,
        primary=primary,
        themes=themes,
        centroids=centroids,
        backend=backend,
        vectors=vectors,
        embedder=embedder,
        doc_sentiment=doc_sentiment,
        language=lang.code,
    )


def merge_themes(model: ThemeModel, merges: dict[str, str]) -> ThemeModel:
    """Fold themes into others (``{"T07": "T03"}`` moves T07 into T03), keeping ids stable.

    Used to apply duplicate suggestions from the LLM review or from a human analyst.
    """
    cluster_of = dict(zip(model.themes["theme_id"], model.themes["cluster"], strict=True))
    parent = {
        cluster_of[a]: cluster_of[b]
        for a, b in merges.items()
        if a in cluster_of and b in cluster_of and a != b
    }

    def root(c: int) -> int:
        seen = set()
        while c in parent and c not in seen:  # follow chains, stop on cycles
            seen.add(c)
            c = parent[c]
        return c

    if not parent:
        return model
    segments = model.segments.copy()
    segments["cluster"] = [root(c) if c != OTHER else c for c in segments["cluster"]]
    return _finalize(
        segments,
        model.vectors,
        model.doc_sentiment,
        len(model.primary),
        model.backend,
        model.embedder,
        get_language(model.language),
    )


def _theme_metrics(mentions: pd.DataFrame, n_docs: int) -> pd.DataFrame:
    grouped = mentions.groupby("theme_id")

    def mix(col: str) -> pd.Series:
        # built by hand: groupby.apply would expand the returned dicts into a MultiIndex
        values = {tid: s.value_counts(normalize=True).round(3).to_dict() for tid, s in grouped[col]}
        return pd.Series(values, dtype=object)

    unique_accounts = mentions.drop_duplicates(["theme_id", "account_id"])
    # Revenue-weighted demand: every account "votes" with its MRR, split evenly across all
    # of its mentions. Summing raw MRR of every account that ever mentioned a theme would
    # saturate: a big customer that complains about everything would dominate every theme.
    per_account = mentions.groupby("account_id")["doc"].transform("size")
    weighted = (mentions["mrr"] / per_account).groupby(mentions["theme_id"]).sum()
    metrics = pd.DataFrame(
        {
            "mentions": grouped["doc"].nunique(),
            "primary_mentions": grouped["primary"].sum(),
            "accounts": grouped["account_id"].nunique(),
            "mrr_exposed": unique_accounts.groupby("theme_id")["mrr"].sum(),
            "mrr_weighted": weighted,
            "mean_sentiment": grouped["sentiment"].mean(),
            "negative_share": grouped["sentiment"].apply(lambda s: float((s < NEGATIVE).mean())),
            "first_seen": grouped["created_at"].min(),
            "last_seen": grouped["created_at"].max(),
            "plan_mix": mix("plan"),
            "channel_mix": mix("channel"),
        }
    )
    metrics.index.name = "theme_id"
    metrics["share"] = metrics["mentions"] / max(n_docs, 1)
    metrics["arr_exposed"] = metrics["mrr_exposed"] * 12
    return metrics.reset_index()


def analyze(
    feedback: pd.DataFrame | str,
    accounts: pd.DataFrame | str | None = None,
    releases: pd.DataFrame | str | None = None,
    config: Config = Config(),
    as_of: str | pd.Timestamp | None = None,
    model: ThemeModel | None = None,
    embedder: Embedder | None = None,
) -> Analysis:
    fb = load_feedback(feedback)
    acc = load_accounts(accounts)
    rel = load_releases(releases)
    if config.redact_pii:  # before anything is embedded, stored in a report or sent to an API
        fb["text"] = fb["text"].map(redact)
    fb["sentiment"] = score_feedback(fb, lang=get_language(config.language))
    if model is None:
        model = build_theme_model(fb, config, embedder=embedder, sentiment=fb["sentiment"])

    if acc is not None:
        extra = [c for c in ("plan", "mrr", "company", "seats") if c in acc]
        fb = fb.merge(acc[["account_id", *extra]], on="account_id", how="left")
        fb["mrr"] = fb["mrr"].fillna(0.0)
        fb["plan"] = fb["plan"].fillna("unknown")
    else:
        fb["mrr"], fb["plan"] = 0.0, "unknown"
    fb["theme_id"] = model.primary

    as_of = pd.Timestamp(as_of) if as_of is not None else fb["created_at"].max()
    visible = fb["created_at"] < as_of.normalize() + pd.Timedelta(days=1)
    mentions = model.doc_themes.merge(
        fb[["created_at", "account_id", "mrr", "plan", "channel", "sentiment"]],
        left_on="doc",
        right_index=True,
    )
    mentions = mentions[mentions["created_at"] < as_of.normalize() + pd.Timedelta(days=1)]
    dates = fb.loc[visible, "created_at"]

    base = model.themes.drop(columns=["cluster"])
    window_start = as_of.normalize() + pd.Timedelta(days=1 - config.score_window_days)
    in_window = mentions["created_at"] >= window_start
    n_window = int((dates >= window_start).sum())
    metrics = _theme_metrics(mentions[in_window], n_window)
    themes = base.merge(metrics, on="theme_id", how="left")
    all_time = mentions.groupby("theme_id")["doc"].nunique().rename("mentions_all_time")
    themes = themes.merge(all_time, on="theme_id", how="left")
    fill = {
        "mentions": 0,
        "mentions_all_time": 0,
        "primary_mentions": 0,
        "accounts": 0,
        "mrr_exposed": 0.0,
        "mrr_weighted": 0.0,
        "arr_exposed": 0.0,
        "negative_share": 0.0,
        "share": 0.0,
    }
    themes = themes.fillna(fill)
    theme_ids = themes["theme_id"].tolist()
    trends = detect_trends(
        mentions,
        dates,
        theme_ids,
        as_of,
        config.recent_days,
        config.baseline_days,
        config.alpha,
        normalization=config.volume_normalization,
    )
    themes = themes.merge(trends, on="theme_id", how="left")

    has_revenue = acc is not None and float(fb["mrr"].sum()) > 0
    weights = config.weights if has_revenue else replace(config.weights, revenue=0.0)
    themes = score_themes(themes, weights)

    radar = effects = None
    if rel is not None and len(rel):
        rel = rel[rel["date"] <= as_of]
        texts = (rel["title"] + ". " + rel["description"]).tolist()
        vectors, centroids = model.embedder.encode(texts), model.centroids
        semantic = getattr(model.embedder, "semantic_part", None)
        if semantic is not None:  # hybrid: release notes are written in product language,
            # not the customers' words, so match on meaning and ignore incidental word overlap
            vectors, centroids = _unit(semantic(vectors)), _unit(semantic(centroids))
        matched = match_releases(
            rel, vectors, model.themes, centroids, config.release_match_similarity
        )
        radar, effects = release_radar(
            matched,
            mentions,
            dates,
            theme_ids,
            config.impact_window_days,
            config.alpha,
            normalization=config.volume_normalization,
        )

    return Analysis(
        feedback=fb,
        mentions=mentions,
        themes=themes.reset_index(drop=True),
        weekly=weekly_mentions(mentions, dates),
        releases=radar,
        side_effects=effects,
        model=model,
        config=config,
        as_of=as_of,
        has_revenue=has_revenue,
        inputs={"feedback": feedback, "accounts": accounts, "releases": releases},
    )


def rerun(
    analysis: Analysis,
    model: ThemeModel | None = None,
    as_of: str | pd.Timestamp | None = None,
    config: Config | None = None,
) -> Analysis:
    """Re-analyze the same inputs with a modified model, date or config (no re-embedding)."""
    return analyze(
        **analysis.inputs,
        config=config or analysis.config,
        as_of=as_of if as_of is not None else analysis.as_of,
        model=model or analysis.model,
    )


def review_with_claude(analysis: Analysis, analyst=None) -> Analysis:
    """Let Claude name, classify and de-duplicate the themes, then re-analyze.

    Raises :class:`clamor.llm.LLMUnavailable` if Claude cannot be reached.
    """
    from . import llm

    analyst = analyst or llm.ClaudeAnalyst()
    review = analyst.review_themes(analysis.themes, language=analysis.config.output_language)
    model = merge_themes(analysis.model, llm.merge_suggestions(review))
    model.themes = llm.apply_review(model.themes, review)
    return rerun(analysis, model=model)


def rescore(analysis: Analysis, weights: Weights) -> pd.DataFrame:
    """Re-rank with different weights without recomputing anything else."""
    if not analysis.has_revenue:
        weights = replace(weights, revenue=0.0)
    return score_themes(analysis.themes, weights)
