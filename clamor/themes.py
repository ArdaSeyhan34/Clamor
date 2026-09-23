"""Discovering themes: clustering content segments and describing each cluster.

Why agglomerative clustering with a distance threshold instead of k-means?

* The number of themes is unknown and is exactly what we are trying to find.
* Feedback themes are very unbalanced (hundreds of "dark mode" requests next to a dozen
  SSO requests). K-means prefers similar-sized clusters: it splits big themes and merges
  small ones, and small themes are often the expensive ones.
* A cosine-distance threshold has a meaning a PM can reason about: "how similar do two
  comments have to be to count as the same request?"

For large inputs, segments are first compressed into micro-clusters with k-means so the
quadratic agglomerative step stays tractable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from .lang import ENGLISH, LanguagePack

OTHER = -1


def _vectorizer_options(lang: LanguagePack, keywords: bool = False) -> dict:
    """CountVectorizer settings per language (English keeps its original settings)."""
    if lang.code == "en":
        opts: dict = {"stop_words": "english"}
        if keywords:
            opts["token_pattern"] = r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b"
        return opts
    return {
        "stop_words": sorted(lang.stop_words),
        "preprocessor": lang.lower,  # Turkish-aware lower-casing (İ -> i, I -> ı)
        "token_pattern": r"(?u)\b[^\W\d_][^\W\d_\-]+\b",
    }


KIND_CUES = ENGLISH.kind_cues  # backwards-compatible alias


@dataclass
class ThemeModel:
    """Everything learned from the text; independent of revenue, dates and weights."""

    segments: pd.DataFrame  # doc, position, text, is_boilerplate, theme
    doc_themes: pd.DataFrame  # one row per (doc, theme) mention with `primary` flag
    primary: np.ndarray  # primary theme per doc (OTHER if none)
    themes: pd.DataFrame  # theme_id, label, keywords, headline, kind, examples, ...
    centroids: np.ndarray  # one normalized vector per theme, aligned with `themes`
    backend: str
    vectors: np.ndarray | None = None  # one vector per segment
    embedder: object | None = None  # kept to embed release notes in the same space
    doc_sentiment: np.ndarray | None = None
    language: str = "en"


def cluster_vectors(
    vectors: np.ndarray,
    distance_threshold: float = 0.7,
    min_size: int = 5,
    reassign_similarity: float = 0.5,
    max_direct: int = 6000,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster unit vectors; returns labels sorted by cluster size (0 = largest).

    Clusters smaller than `min_size` are dissolved: their members join the most similar
    large cluster if that similarity is at least `reassign_similarity`, otherwise they are
    labeled ``OTHER`` (-1).
    """
    n = len(vectors)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n < 3:
        return np.zeros(n, dtype=int)

    if n > max_direct:
        n_micro = min(max_direct // 2, n // 3)
        micro = KMeans(n_micro, n_init=1, random_state=random_state).fit(vectors)
        centers = micro.cluster_centers_
        centers = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
        macro = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        ).fit(centers)
        raw = macro.labels_[micro.labels_]
    else:
        raw = (
            AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
            .fit(vectors)
            .labels_
        )

    sizes = np.bincount(raw)
    big = np.flatnonzero(sizes >= min_size)
    if len(big) == 0:
        return np.full(n, OTHER)
    centroids = np.vstack([vectors[raw == c].mean(axis=0) for c in big])
    centroids /= np.clip(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12, None)
    sim = vectors @ centroids.T
    nearest = big[sim.argmax(axis=1)]
    labels = np.where(
        np.isin(raw, big), raw, np.where(sim.max(axis=1) >= reassign_similarity, nearest, OTHER)
    )
    # relabel 0..k-1 by descending size so theme ids are stable and meaningful
    kept = [c for c in pd.Series(labels[labels != OTHER]).value_counts().index]
    mapping = {old: new for new, old in enumerate(kept)}
    return np.array([mapping.get(x, OTHER) for x in labels])


def consolidate(
    labels: np.ndarray,
    texts: list[str],
    semantic_vectors: np.ndarray,
    min_semantic: float = 0.45,
    min_lexical: float = 0.18,
    lang: LanguagePack = ENGLISH,
) -> np.ndarray:
    """Merge clusters that are near-duplicates in *both* meaning and vocabulary.

    Clustering at a strict threshold gives very pure but sometimes over-split themes
    ("SSO" vs "SAML single sign-on"). Neither signal alone is a safe merge criterion: by
    meaning alone "Slack integration" sits close to "email notifications"; by vocabulary
    alone two unrelated complaints can share generic words. Requiring both to agree merges
    true duplicates and leaves distinct themes alone. Merges are transitive.
    """
    ids = sorted(set(labels) - {OTHER})
    if len(ids) < 2:
        return labels
    cents = np.vstack([semantic_vectors[labels == c].mean(axis=0) for c in ids])
    cents /= np.clip(np.linalg.norm(cents, axis=1, keepdims=True), 1e-12, None)
    semantic = cents @ cents.T
    docs = [" ".join(t for t, lab in zip(texts, labels, strict=True) if lab == c) for c in ids]
    lex = TfidfTransformer(sublinear_tf=True).fit_transform(
        CountVectorizer(**_vectorizer_options(lang)).fit_transform(docs)
    )
    lexical = (lex @ lex.T).toarray()

    ok = (semantic >= min_semantic) & (lexical >= min_lexical)
    # complete linkage: two groups merge only if *every* pair across them qualifies, so a
    # chain A~B~C cannot glue two unrelated themes together through a middle one
    groups = [{i} for i in range(len(ids))]
    candidates = sorted(
        (
            (semantic[i, j], i, j)
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
            if ok[i, j]
        ),
        reverse=True,
    )
    for _, i, j in candidates:
        gi = next(g for g in groups if i in g)
        gj = next(g for g in groups if j in g)
        if gi is not gj and all(ok[a, b] for a in gi for b in gj):
            gi |= gj
            groups.remove(gj)
    root = {ids[i]: ids[min(g)] for g in groups for i in g}
    merged = np.array([root.get(x, OTHER) for x in labels])
    order = pd.Series(merged[merged != OTHER]).value_counts().index
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping.get(x, OTHER) for x in merged])


def keywords_by_cluster(
    texts: list[str], labels: np.ndarray, top_n: int = 6, lang: LanguagePack = ENGLISH
) -> dict[int, list]:
    """Class-based TF-IDF (c-TF-IDF): which n-grams are distinctive for each cluster."""
    clusters = sorted(set(labels) - {OTHER})
    if not clusters:
        return {}
    docs = [" ".join(t for t, lab in zip(texts, labels, strict=True) if lab == c) for c in clusters]
    vec = CountVectorizer(ngram_range=(1, 2), min_df=1, **_vectorizer_options(lang, True))
    counts = vec.fit_transform(docs)
    weights = TfidfTransformer(sublinear_tf=True).fit_transform(counts).toarray()
    vocab = np.array(vec.get_feature_names_out())
    # prefer phrases: "dark mode" says more than "dark" and "mode" separately
    weights = weights * np.where(np.char.count(vocab.astype(str), " ") > 0, 1.6, 1.0)
    result = {}
    for row, c in enumerate(clusters):
        chosen: list[str] = []
        for idx in np.argsort(-weights[row]):
            term = vocab[idx]
            if weights[row, idx] <= 0:
                break
            # skip a unigram already covered by a chosen bigram, and vice versa
            if any(term in c_ or c_ in term for c_ in chosen):
                continue
            chosen.append(term)
            if len(chosen) == top_n:
                break
        result[c] = chosen
    return result


def classify_kind(texts: list[str], mean_sentiment: float, lang: LanguagePack = ENGLISH) -> str:
    """Heuristic theme type from cue phrases; the LLM layer can override it."""
    if not texts:
        return "other"
    joined = [lang.normalize(t) for t in texts]
    rates = {
        kind: np.mean([any(re.search(p, t) for p in pats) for t in joined])
        for kind, pats in lang.kind_cues.items()
    }
    if rates["pricing"] >= 0.35:
        return "pricing"
    if mean_sentiment >= 0.35 and rates["bug"] < 0.2:
        return "praise"
    if rates["bug"] >= max(0.3, rates["feature_request"]):
        return "bug"
    # weaker malfunction cues still mean "bug" when the tone is clearly negative
    if rates["bug"] >= max(0.2, rates["feature_request"]) and mean_sentiment < -0.2:
        return "bug"
    if rates["feature_request"] >= 0.3:
        return "feature_request"
    return "ux"


def offline_name(headline: str, max_len: int = 60) -> str:
    """Default theme name without an LLM: the most representative real quote, trimmed.

    Keyword soup ("users hand · risk need") is hard to read; a customer's own sentence
    ("Offboarding users by hand is a compliance risk") is not. Claude replaces it with a
    proper roadmap-style name when enabled.
    """
    text = headline.strip().rstrip(".!?")
    if len(text) <= max_len:
        return text
    cut = text[:max_len].rsplit(" ", 1)[0].rstrip(",;:")
    return cut + "…"


def _headline(
    texts: list[str], vectors: np.ndarray, centroid: np.ndarray, lang: LanguagePack = ENGLISH
) -> str:
    """A readable, real-user description of the theme.

    Among the most central segments, prefer short ones and the wording customers use most
    often: a sentence that appears many times is unlikely to contain a one-off typo. For
    languages with diacritics, spellings with and without them count as one wording, and
    the properly written variant ("Canlı destek") wins over "canli destek".
    """
    sims = vectors @ centroid
    top = np.argsort(-sims)[:8]
    key = str.lower if lang.code == "en" else lang.normalize
    counts = pd.Series([key(t) for t in texts]).value_counts()

    def quality(t: str) -> tuple:
        if lang.code == "en":
            return ()
        return (-sum(not c.isascii() for c in t), -sum(c.isupper() for c in t))

    best = min(
        top,
        key=lambda i: (len(texts[i]) > 90, -counts[key(texts[i])], *quality(texts[i]), -sims[i]),
    )
    if lang.code != "en":  # the best-spelled variant of that wording anywhere in the theme
        same = [t for t in texts if key(t) == key(texts[best])]
        return min(same, key=quality).rstrip(".")
    return texts[best].rstrip(".")


def describe_themes(
    segments: pd.DataFrame,
    vectors: np.ndarray,
    labels: np.ndarray,
    doc_sentiment: np.ndarray,
    n_examples: int = 8,
    lang: LanguagePack = ENGLISH,
) -> tuple[pd.DataFrame, np.ndarray]:
    texts = segments["text"].tolist()
    keywords = keywords_by_cluster(texts, labels, lang=lang)
    rows, centroids = [], []
    for c in sorted(keywords):
        member = np.flatnonzero(labels == c)
        centroid = vectors[member].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        centroids.append(centroid)
        member_texts = [texts[i] for i in member]
        docs = segments["doc"].to_numpy()[member]
        sims = vectors[member] @ centroid
        order = member[np.argsort(-sims)]
        seen, examples = set(), []
        for i in order:  # distinct, representative quotes
            key = lang.normalize(texts[i])
            if key not in seen:
                seen.add(key)
                examples.append(texts[i])
            if len(examples) == n_examples:
                break
        mean_sent = float(np.mean(doc_sentiment[np.unique(docs)]))
        kw = keywords[c]
        headline = _headline(member_texts, vectors[member], centroid, lang)
        rows.append(
            {
                "theme_id": f"T{c + 1:02d}",
                "cluster": c,
                "name": offline_name(headline),
                "summary": "",
                "label": " · ".join(kw[:3]),
                "keywords": kw,
                "headline": headline,
                "kind": classify_kind(member_texts, mean_sent, lang),
                "examples": examples,
                "cohesion": float(np.mean(sims)),
            }
        )
    return pd.DataFrame(rows), np.vstack(centroids) if centroids else np.zeros((0, 1))
