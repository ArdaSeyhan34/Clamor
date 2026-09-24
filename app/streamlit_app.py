"""Clamor dashboard.  Run with:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run from a fresh clone without installing the package

from clamor import charts, i18n, llm, synth  # noqa: E402
from clamor.briefs import theme_evidence, write_brief  # noqa: E402
from clamor.config import PRESETS, Config, Weights  # noqa: E402
from clamor.embeddings import get_embedder  # noqa: E402
from clamor.evaluate import evaluate_all  # noqa: E402
from clamor.insights import headline_insights  # noqa: E402
from clamor.io import (  # noqa: E402
    combine_feedback,
    load_accounts,
    load_feedback,
    load_releases,
    read_table,
)
from clamor.lang import get_language  # noqa: E402
from clamor.pipeline import analyze, build_theme_model, merge_themes  # noqa: E402
from clamor.privacy import redact  # noqa: E402
from clamor.report import KIND, release_table, roadmap_table, status_label  # noqa: E402
from clamor.scoring import contributions  # noqa: E402
from clamor.sentiment import score_feedback  # noqa: E402

st.set_page_config(page_title="Clamor", page_icon="\U0001f4e3", layout="wide")

try:  # Streamlit Cloud secrets -> environment, so the SDK finds the key
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ.setdefault("ANTHROPIC_API_KEY", st.secrets["ANTHROPIC_API_KEY"])
except Exception:  # no secrets file locally
    pass

MODE = "dark" if getattr(getattr(st.context, "theme", None), "type", "light") == "dark" else "light"
DEMOS = {  # scenario -> (label, data folder, language, product names, early-warning tip)
    "tempo": (
        "Demo: Tempo (English B2B SaaS)",
        ROOT / "data" / "demo",
        "en",
        ("Tempo",),
        "Tip: move **Analyze as of** in the sidebar to mid-March to watch the sync "
        "regression get flagged.",
    ),
    "lezzo": (
        "Demo: Lezzo (Turkish meal-card app)",
        ROOT / "data" / "demo_lezzo",
        "tr",
        ("Lezzo",),
        "Tip: move **Analyze as of** in the sidebar to late March to watch the login "
        "problems after v5.1 get flagged.",
    ),
}
UPLOAD = "upload"
LANGUAGES = {"English": "en", "Türkçe": "tr"}
EVAL_COLUMNS = {  # evaluation tables: column -> heading, per interface language
    "tr": {
        "spikes_detected": "yakalanan sıçrama",
        "spikes_total": "toplam sıçrama",
        "median_days_to_detect": "yakalama süresi (medyan gün)",
        "false_alarm_episodes": "yanlış alarm",
        "false_alarm_days": "yanlış alarm günü",
        "spurious_trend_days": "yersiz eğilim günü",
        "spike": "sıçrama",
        "started": "başlangıç",
        "method": "yöntem",
        "first_alert": "ilk uyarı",
        "days_to_detect": "yakalama süresi (gün)",
        "true_theme": "gerçek tema",
        "items": "öğe",
        "recovered_share": "doğru temaya düşen pay",
        "themes": "tema sayısı",
        "largest_theme": "en büyük tema",
        "largest_theme_share": "en büyük temanın payı",
        "largest_theme_purity": "en büyük temanın saflığı",
    }
}


def interface_language() -> str:
    """?lang=tr in the URL, else the browser's language, else English."""
    requested = st.query_params.get("lang", "")
    if requested in i18n.OUTPUT_LANGUAGES:
        return requested
    locale = str(getattr(st.context, "locale", None) or "")
    return "tr" if locale.lower().startswith("tr") else "en"


st.sidebar.title("\U0001f4e3 Clamor")
UI = st.sidebar.segmented_control(
    "Interface language / Arayüz dili",
    list(LANGUAGES),
    default=next(k for k, v in LANGUAGES.items() if v == interface_language()),
    label_visibility="collapsed",
)
UI = LANGUAGES.get(UI) or interface_language()  # None when the selection is clicked again
st.query_params["lang"] = UI


def _(text: str, **values) -> str:
    """Translate dashboard text into the interface language (see clamor.i18n)."""
    return i18n.t(text, UI, **values)


# --------------------------------------------------------------------------- data & models
@st.cache_data(show_spinner=False)
def load_demo(scenario: str, folder: str) -> dict:
    folder = Path(folder)
    if not (folder / "feedback.csv").exists():
        synth.generate_scenario(scenario).save(folder)
    ds = synth.SyntheticDataset.load(folder)
    return {name: getattr(ds, name) for name in ds.FILES}


def read_upload(upload) -> pd.DataFrame | None:
    """Read an uploaded export with Clamor's loader (encoding, delimiter, Excel)."""
    if upload is None:
        return None
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix) as tmp:
        tmp.write(upload.getvalue())
        tmp.flush()
        return read_table(tmp.name)


@st.cache_resource(show_spinner=False)
def embedder_for(backend: str, language: str):
    return get_embedder(backend, lang=get_language(language))


@st.cache_resource(show_spinner=_("Discovering themes (embedding and clustering)..."))
def theme_model(
    data_key: str, _feedback: pd.DataFrame, backend: str, language: str, product: tuple[str, ...]
):
    fb = load_feedback(_feedback)
    cfg = Config(language=language, embedding=backend, product_names=product)
    if cfg.redact_pii:
        fb["text"] = fb["text"].map(redact)
    return build_theme_model(
        fb,
        cfg,
        embedder=embedder_for(backend, language),
        sentiment=score_feedback(fb, lang=get_language(language)),
    )


@st.cache_resource(show_spinner=_("Claude is reviewing the themes..."))
def claude_reviewed_model(data_key: str, _model, _themes: pd.DataFrame, language: str):
    review = llm.ClaudeAnalyst().review_themes(_themes, language=language)
    model = merge_themes(_model, llm.merge_suggestions(review))
    return replace(model, themes=llm.apply_review(model.themes, review))  # keep cache intact


@st.cache_data(show_spinner=_("Backtesting early warnings day by day..."))
def cached_evaluation(data_key: str, _analysis, _dataset: dict) -> dict:
    return evaluate_all(_analysis, synth.SyntheticDataset(**_dataset))


def md(text: str) -> str:
    """Escape dollar signs so Streamlit does not render "$9k ... $5k" as LaTeX."""
    return text.replace("$", "\\$")


def frame_key(*frames: pd.DataFrame | None) -> str:
    h = hashlib.sha1()
    for f in frames:
        if f is not None:
            h.update(pd.util.hash_pandas_object(f, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- sidebar
st.sidebar.caption(_("Customer feedback → prioritized, evidence-backed roadmap"))

source = st.sidebar.radio(
    _("Data"),
    [*DEMOS, UPLOAD],
    format_func=lambda k: _(DEMOS[k][0]) if k in DEMOS else _("Upload your own"),
    label_visibility="collapsed",
)
dataset = None
tip = None
if source in DEMOS:
    _label, folder, language, product_names, tip = DEMOS[source]
    dataset = load_demo(source, str(folder))
    feedback_raw, accounts_raw, releases_raw = (
        dataset["feedback"],
        dataset["accounts"],
        dataset["releases"],
    )
else:
    kinds = ["csv", "xlsx", "json"]
    up_fb = st.sidebar.file_uploader(
        _("Feedback: one or more exports (text + date required)"),
        type=kinds,
        accept_multiple_files=True,
    )
    up_acc = st.sidebar.file_uploader(_("Accounts (account_id + mrr, optional)"), type=kinds)
    up_rel = st.sidebar.file_uploader(_("Releases (date + title, optional)"), type=kinds)
    language = LANGUAGES[
        st.sidebar.selectbox(
            _("Language of the feedback"),
            list(LANGUAGES),
            index=list(LANGUAGES.values()).index(UI),
        )
    ]
    names = st.sidebar.text_input(_("Product name(s) to ignore, comma separated"), "")
    product_names = tuple(n.strip() for n in names.split(",") if n.strip())
    if not up_fb:
        st.title(_("Bring your own feedback"))
        st.markdown(
            _(
                "Upload a CSV export from your support tool, app store reviews or NPS survey. "
                "Only a **text** column and a **date** column are required; common names such "
                "as `body`, `comment`, `review`, `created`, `timestamp` (and Turkish ones such "
                "as `Yorum`, `Açıklama`, `Tarih`, `Puan`) are recognized automatically, as are "
                "Google Play Console exports. Several files (say, store reviews and support "
                "tickets) are combined, each keeping its own channel. Phone numbers, e-mails, "
                "card numbers, IBANs and national ID numbers are masked before anything is "
                "analyzed."
            )
            + "\n\n"
            + _(
                "Add an **accounts** file (`account_id`, `mrr`, `plan`) to weigh themes by "
                "revenue, and a **releases** file (`date`, `title`, `description`) to get the "
                "release radar."
            )
            + "\n\n"
            + _(
                "Files are processed in memory by the server running this app and are not "
                "stored. For confidential data, run the app on your own machine: then nothing "
                "leaves it except, if you enable it, theme summaries sent to Claude."
            )
        )
        st.stop()
    frames = [read_upload(f) for f in up_fb]
    feedback_raw = (
        frames[0]
        if len(frames) == 1
        else combine_feedback(frames, names=[Path(f.name).stem for f in up_fb])
    )
    accounts_raw = read_upload(up_acc)
    releases_raw = read_upload(up_rel)

try:
    fb_valid = load_feedback(feedback_raw)
    acc_valid = load_accounts(accounts_raw)
    load_releases(releases_raw)  # validate early, with a friendly error
except ValueError as exc:
    st.error(_("Could not read the input: {error}", error=exc))
    st.stop()

min_day, max_day = fb_valid["created_at"].min().date(), fb_valid["created_at"].max().date()
first_day = min_day + dt.timedelta(days=28)
try:  # shareable links: ?as_of=2026-03-20 opens the dashboard on that day
    requested = dt.date.fromisoformat(st.query_params.get("as_of", ""))
    start_value = min(max(requested, first_day), max_day)
except ValueError:
    start_value = max_day
as_of = st.sidebar.slider(
    _("Analyze as of"),
    min_value=first_day,
    max_value=max_day,
    value=start_value,
    format="D.MM.YYYY" if UI == "tr" else "MMM D, YYYY",
    help=_("Time travel: see what Clamor would have told you on that day."),
)

st.sidebar.subheader(_("What matters most?"))
preset = st.sidebar.selectbox(
    _("Weight preset"), list(PRESETS), index=0, format_func=lambda p: _(p).capitalize()
)
base = PRESETS[preset]
with st.sidebar.expander(_("Fine-tune weights")):
    weights = Weights(
        reach=st.slider(_("Reach (accounts)"), 0.0, 1.0, base.reach, 0.05),
        revenue=st.slider(_("Revenue-weighted demand"), 0.0, 1.0, base.revenue, 0.05),
        severity=st.slider(_("Severity (negative sentiment)"), 0.0, 1.0, base.severity, 0.05),
        momentum=st.slider(_("Momentum (significant growth)"), 0.0, 1.0, base.momentum, 0.05),
    )

with st.sidebar.expander(_("Model")):
    semantic = "minilm" if language == "en" else "multilingual"
    default = get_language(language).default_embedding
    backend = st.selectbox(
        _("Embedding backend"),
        list(dict.fromkeys([default, semantic, "hybrid", "tfidf"])),
        help=_(
            "Default for this language: {default}. {semantic} = semantic sentence embeddings; "
            "hybrid adds TF-IDF vocabulary; tfidf needs no model download.",
            default=default,
            semantic=semantic,
        ),
    )
    use_claude = st.toggle(
        _("Review themes with Claude"),
        value=False,
        disabled=not llm.available(),
        help=_("Names, classifies and de-duplicates themes. Needs ANTHROPIC_API_KEY."),
    )
    if not llm.available():
        st.caption(_("Set `ANTHROPIC_API_KEY` to enable the Claude analyst layer."))

# --------------------------------------------------------------------------- analysis
data_key = frame_key(feedback_raw) + backend + language + ",".join(product_names)
model = theme_model(data_key, feedback_raw, backend, language, product_names)
config = Config(
    language=language,
    embedding=backend,
    product_names=product_names,
    weights=weights,
    report_language=UI,
)
if use_claude:
    try:
        first = analyze(feedback_raw, accounts_raw, releases_raw, config=config, model=model)
        model = claude_reviewed_model(data_key, model, first.themes, UI)
    except llm.LLMUnavailable as exc:
        st.sidebar.warning(_("Claude unavailable: {error}", error=exc))
analysis = analyze(
    feedback_raw, accounts_raw, releases_raw, config=config, as_of=pd.Timestamp(as_of), model=model
)
themes = analysis.themes
names = dict(zip(themes["theme_id"], themes["name"], strict=True))
road = analysis.roadmap

if model.backend != backend:
    st.warning(
        _(
            "The `{backend}` model could not be loaded, so Clamor fell back to `{fallback}`. "
            "Results will be less accurate.",
            backend=backend,
            fallback=model.backend,
        )
    )

# --------------------------------------------------------------------------- header
st.title(_("What should we build next?"))
fb = analysis.feedback[analysis.feedback["created_at"] < analysis.as_of + pd.Timedelta(days=1)]
people_title = i18n.people(analysis.people, UI, "title")
cols = st.columns(5)
cols[0].metric(_("Feedback items"), i18n.number(len(fb), UI))
cols[1].metric(people_title, i18n.number(fb["account_id"].nunique(), UI))
cols[2].metric(_("Themes"), len(themes))
cols[3].metric(_("Early warnings"), int(themes["status"].isin(["new", "emerging"]).sum()))
if analysis.has_revenue:
    cols[4].metric(
        _("MRR represented"), i18n.money(fb.drop_duplicates("account_id")["mrr"].sum(), UI)
    )
else:
    cols[4].metric(_("Revenue data"), _("not provided"))

tab_road, tab_theme, tab_warn, tab_rel, tab_eval = st.tabs(
    [_("Roadmap"), _("Theme explorer"), _("Early warning"), _("Release radar"), _("Model quality")]
)

# --------------------------------------------------------------------------- roadmap
with tab_road:
    insight_cols = st.columns(2)
    for i, ins in enumerate(headline_insights(analysis, lang=UI)):
        with insight_cols[i % 2].container(border=True):
            st.markdown(md(f"**{ins.title}**  \n{ins.detail}"))

    st.subheader(_("Priority ranking"))
    w = weights.normalized()
    st.caption(
        _(
            "Points each signal contributes. Weights: reach {reach}, revenue {revenue}, "
            "severity {severity}, momentum {momentum}. Reach, revenue and severity use the "
            "last {days} days.",
            reach=i18n.pct(w.reach, UI),
            revenue=i18n.pct(w.revenue, UI),
            severity=i18n.pct(w.severity, UI),
            momentum=i18n.pct(w.momentum, UI),
            days=config.score_window_days,
        )
    )
    parts = contributions(
        themes, weights if analysis.has_revenue else replace(weights, revenue=0.0)
    )
    st.plotly_chart(charts.priority_chart(themes, parts, mode=MODE, lang=UI), theme=None)
    score_col = _("Score")
    st.dataframe(
        roadmap_table(analysis, lang=UI),
        hide_index=True,
        column_config={
            score_col: st.column_config.ProgressColumn(
                score_col, min_value=0, max_value=100, format="%.1f"
            )
        },
    )

    st.subheader(_("Counting votes vs. weighing evidence"))
    st.caption(
        _(
            "Left: rank by raw number of mentions. Right: Clamor's priority. Highlighted "
            "themes moved the most."
        )
    )
    st.plotly_chart(charts.rank_shift_chart(themes, mode=MODE, lang=UI), theme=None)

# --------------------------------------------------------------------------- theme explorer
with tab_theme:
    ranked = list(road["theme_id"])
    options = ranked + [t for t in themes["theme_id"] if t not in ranked]
    pick = st.selectbox(_("Theme"), options, format_func=lambda t: f"{t} · {names[t]}")
    row = themes.set_index("theme_id").loc[pick]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### {row['name']}")
        if row.get("summary"):
            st.markdown(row["summary"])
        st.caption(
            _(
                "{kind} · keywords: {keywords}",
                kind=_(KIND.get(row["kind"], row["kind"])),
                keywords=", ".join(row["keywords"][:6]),
            )
        )
        m = st.columns(4)
        m[0].metric(_("Mentions ({days}d)", days=config.score_window_days), int(row["mentions"]))
        m[1].metric(people_title, int(row["accounts"]))
        sentiment = f"{row['mean_sentiment']:+.2f}"
        m[2].metric(_("Sentiment"), sentiment.replace(".", ",") if UI == "tr" else sentiment)
        lift = row["lift"]
        m[3].metric(  # the status is the label: it is too long for a metric value
            status_label(row["status"], UI),
            f"x{i18n.number(lift, UI, 2)}" if np.isfinite(lift) else "–",
            help=_(
                "Trend: mention rate over the last {days} days vs. the baseline.",
                days=config.recent_days,
            ),
        )
        st.plotly_chart(
            charts.timeline_chart(
                analysis.weekly,
                names,
                [pick],
                analysis.releases,
                mode=MODE,
                normalize=False,
                lang=UI,
            ),
            theme=None,
        )
    with right:
        st.markdown(f"**{_('What customers say')}**")
        for q in list(row["examples"])[:6]:
            st.markdown(md(f"> {q}"))
        mix = pd.Series(row["plan_mix"], name=_("share")).sort_values(ascending=False)
        if analysis.has_revenue and len(mix):
            st.markdown(f"**{_('Plan mix')}**")
            st.dataframe(mix.map(lambda x: i18n.pct(x, UI)))

    st.markdown(f"#### {_('Opportunity brief')}")
    brief_llm = st.toggle(
        _("Write it with Claude"), value=False, disabled=not llm.available(), key="brief_llm"
    )
    text, src = write_brief(analysis, pick, use_llm=brief_llm, lang=UI)
    with st.container(border=True):
        st.markdown(md(text))
    st.download_button(_("Download brief (Markdown)"), text, file_name=f"brief-{pick}.md")
    with st.expander(_("Evidence pack sent to the brief writer")):
        st.json(theme_evidence(analysis, pick), expanded=False)

# --------------------------------------------------------------------------- early warning
with tab_warn:
    st.caption(
        _(
            "Each theme's mention rate in the last {recent} days vs. the {baseline} days "
            "before, normalized for overall feedback volume (median-of-ratios), tested with an "
            "exact Poisson rate test and corrected for multiple comparisons "
            "(Benjamini-Hochberg).",
            recent=config.recent_days,
            baseline=config.baseline_days,
        )
        + (f" {_(tip)}" if tip else "")
    )
    warn = themes[
        [
            "theme_id",
            "name",
            "status",
            "recent_mentions",
            "baseline_mentions",
            "lift",
            "lift_ci_low",
            "lift_ci_high",
            "q_value",
        ]
    ].copy()
    warn["order"] = warn["status"].map(
        {"new": 0, "emerging": 1, "rising": 2, "declining": 3, "stable": 4, "quiet": 5}
    )
    warn = warn.sort_values(["order", "lift"], ascending=[True, False]).drop(columns="order")
    warn["status"] = warn["status"].map(lambda s: status_label(s, UI))
    st.dataframe(
        warn,
        hide_index=True,
        column_config={
            "theme_id": "ID",
            "name": _("Theme"),
            "status": _("Status"),
            "recent_mentions": _("Recent"),
            "baseline_mentions": _("Baseline"),
            "lift": st.column_config.NumberColumn(_("Rate vs baseline"), format="x%.2f"),
            "lift_ci_low": st.column_config.NumberColumn(_("95% CI low"), format="x%.2f"),
            "lift_ci_high": st.column_config.NumberColumn(_("95% CI high"), format="x%.2f"),
            "q_value": st.column_config.NumberColumn(_("q-value (FDR)"), format="%.4f"),
        },
    )
    default = [
        t for t in themes.loc[themes["status"].isin(["new", "emerging", "declining"]), "theme_id"]
    ][:4] or list(road["theme_id"][:3])
    chosen = st.multiselect(
        _("Compare themes over time (up to 4)"),
        options,
        default=default,
        max_selections=4,
        format_func=lambda t: f"{t} · {names[t]}",
    )
    st.plotly_chart(
        charts.timeline_chart(
            analysis.weekly, names, chosen, analysis.releases, mode=MODE, lang=UI
        ),
        theme=None,
    )

# --------------------------------------------------------------------------- release radar
with tab_rel:
    if analysis.releases is None or analysis.releases.empty:
        st.info(
            _(
                "Upload a releases file (date, title, description) to see whether shipped "
                "work changed what customers talk about."
            )
        )
    else:
        st.caption(
            _(
                "For each release, the linked theme's mention rate after vs. before (windows "
                "cut at neighbouring releases, normalized for overall volume). Observational "
                "evidence, not a controlled experiment."
            )
        )
        st.plotly_chart(
            charts.release_chart(analysis.releases, names, mode=MODE, lang=UI), theme=None
        )
        st.dataframe(release_table(analysis, lang=UI), hide_index=True)
        if analysis.side_effects is not None and len(analysis.side_effects):
            st.markdown(
                _(
                    "**Suspected side effects**: themes that spiked after a release they "
                    "were not linked to."
                )
            )
            fx = analysis.side_effects.assign(theme=analysis.side_effects["theme_id"].map(names))
            st.dataframe(
                fx[["version", "theme", "pre_mentions", "post_mentions", "rate_ratio", "q_value"]],
                hide_index=True,
                column_config={
                    "version": _("Release"),
                    "theme": _("Theme"),
                    "pre_mentions": _("Before"),
                    "post_mentions": _("After"),
                    "rate_ratio": st.column_config.NumberColumn(_("Rate change"), format="x%.2f"),
                    "q_value": st.column_config.NumberColumn(_("q-value (FDR)"), format="%.4f"),
                },
            )

# --------------------------------------------------------------------------- model quality
with tab_eval:
    if dataset is None:
        st.info(
            _(
                "Model quality is measured on the synthetic dataset, where the true theme of "
                "every item is known. Switch to the demo data to see it."
            )
        )
    elif pd.Timestamp(as_of) < fb_valid["created_at"].max().normalize():
        st.info(_("Move **Analyze as of** to the last day to run the evaluation."))
    else:
        ev = cached_evaluation(data_key, analysis, dataset)
        t, a = ev["themes"], ev["alerts"]
        c = st.columns(4)
        c[0].metric(_("Adjusted Rand index"), i18n.number(t["adjusted_rand"], UI, 3))
        c[1].metric(_("Theme purity (homogeneity)"), i18n.number(t["homogeneity"], UI, 3))
        c[2].metric(_("Sentiment sign accuracy"), i18n.pct(ev["sentiment"]["sign_accuracy"], UI))
        c[3].metric(
            _("Releases linked correctly"),
            f"{int(ev['releases']['correct'].sum())}/{len(ev['releases'])}",
        )
        st.markdown(_("**Early-warning backtest**: history replayed day by day."))
        summary = pd.DataFrame(a).T.rename(
            index={"clamor": "Clamor", "naive_2x": _("Naive: week ≥ 2x average")}
        )
        st.dataframe(summary.rename(columns=EVAL_COLUMNS.get(UI, {})))
        st.dataframe(
            ev["detection"]
            .rename(columns=EVAL_COLUMNS.get(UI, {}))
            .replace({"clamor": "Clamor", "naive_2x": _("Naive: week ≥ 2x average")}),
            hide_index=True,
        )
        st.markdown(f"**{_('How well each true theme was recovered')}**")
        st.dataframe(ev["per_theme"].rename(columns=EVAL_COLUMNS.get(UI, {})), hide_index=True)

st.caption(
    _("Clamor · open source · synthetic demo data")
    + " · [GitHub](https://github.com/ArdaSeyhan34/clamor)"
)
