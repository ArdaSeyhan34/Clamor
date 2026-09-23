"""Clamor dashboard.  Run with:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run from a fresh clone without installing the package

from clamor import charts, llm, synth  # noqa: E402
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
from clamor.report import STATUS_ICON, release_table, roadmap_table  # noqa: E402
from clamor.scoring import contributions  # noqa: E402
from clamor.sentiment import score_feedback  # noqa: E402

st.set_page_config(page_title="Clamor", page_icon="\U0001f4e3", layout="wide")

try:  # Streamlit Cloud secrets -> environment, so the SDK finds the key
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ.setdefault("ANTHROPIC_API_KEY", st.secrets["ANTHROPIC_API_KEY"])
except Exception:  # no secrets file locally
    pass

MODE = "dark" if getattr(getattr(st.context, "theme", None), "type", "light") == "dark" else "light"
DEMOS = {  # label -> (scenario, data folder, language, product names)
    "Demo: Tempo (English B2B SaaS)": ("tempo", ROOT / "data" / "demo", "en", ("Tempo",)),
    "Demo: Lezzo (Turkish meal-card app)": (
        "lezzo",
        ROOT / "data" / "demo_lezzo",
        "tr",
        ("Lezzo",),
    ),
}
LANGUAGES = {"English": "en", "Türkçe": "tr"}


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


@st.cache_resource(show_spinner="Discovering themes (embedding and clustering)...")
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


@st.cache_resource(show_spinner="Claude is reviewing the themes...")
def claude_reviewed_model(data_key: str, _model, _themes: pd.DataFrame):
    review = llm.ClaudeAnalyst().review_themes(_themes)
    model = merge_themes(_model, llm.merge_suggestions(review))
    return replace(model, themes=llm.apply_review(model.themes, review))  # keep cache intact


@st.cache_data(show_spinner="Backtesting early warnings day by day...")
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
st.sidebar.title("\U0001f4e3 Clamor")
st.sidebar.caption("Customer feedback → prioritized, evidence-backed roadmap")

source = st.sidebar.radio("Data", [*DEMOS, "Upload your own"], label_visibility="collapsed")
dataset = None
if source in DEMOS:
    scenario, folder, language, product_names = DEMOS[source]
    dataset = load_demo(scenario, str(folder))
    feedback_raw, accounts_raw, releases_raw = (
        dataset["feedback"],
        dataset["accounts"],
        dataset["releases"],
    )
else:
    kinds = ["csv", "xlsx", "json"]
    up_fb = st.sidebar.file_uploader(
        "Feedback: one or more exports (text + date required)",
        type=kinds,
        accept_multiple_files=True,
    )
    up_acc = st.sidebar.file_uploader("Accounts (account_id + mrr, optional)", type=kinds)
    up_rel = st.sidebar.file_uploader("Releases (date + title, optional)", type=kinds)
    language = LANGUAGES[st.sidebar.selectbox("Language of the feedback", list(LANGUAGES))]
    names = st.sidebar.text_input("Product name(s) to ignore, comma separated", "")
    product_names = tuple(n.strip() for n in names.split(",") if n.strip())
    if not up_fb:
        st.title("Bring your own feedback")
        st.markdown(
            "Upload a CSV export from your support tool, app store reviews or NPS survey. "
            "Only a **text** column and a **date** column are required; common names such as "
            "`body`, `comment`, `review`, `created`, `timestamp` (and Turkish ones such as "
            "`Yorum`, `Açıklama`, `Tarih`, `Puan`) are recognized automatically, as are Google "
            "Play Console exports. Several files (say, store reviews and support tickets) are "
            "combined, each keeping its own channel. Phone numbers, e-mails, card numbers, "
            "IBANs and national ID numbers are masked before anything is analyzed."
            "\n\nAdd an **accounts** file (`account_id`, `mrr`, `plan`) to weigh themes by "
            "revenue, and a **releases** file (`date`, `title`, `description`) to get the "
            "release radar.\n\nFiles are processed in memory by the server running this app "
            "and are not stored. For confidential data, run the app on your own machine: "
            "then nothing leaves it except, if you enable it, theme summaries sent to Claude."
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
    st.error(f"Could not read the input: {exc}")
    st.stop()

min_day, max_day = fb_valid["created_at"].min().date(), fb_valid["created_at"].max().date()
first_day = min_day + dt.timedelta(days=28)
try:  # shareable links: ?as_of=2026-03-20 opens the dashboard on that day
    requested = dt.date.fromisoformat(st.query_params.get("as_of", ""))
    start_value = min(max(requested, first_day), max_day)
except ValueError:
    start_value = max_day
as_of = st.sidebar.slider(
    "Analyze as of",
    min_value=first_day,
    max_value=max_day,
    value=start_value,
    format="MMM D, YYYY",
    help="Time travel: see what Clamor would have told you on that day.",
)

st.sidebar.subheader("What matters most?")
preset = st.sidebar.selectbox("Weight preset", list(PRESETS), index=0, format_func=str.capitalize)
base = PRESETS[preset]
with st.sidebar.expander("Fine-tune weights"):
    weights = Weights(
        reach=st.slider("Reach (accounts)", 0.0, 1.0, base.reach, 0.05),
        revenue=st.slider("Revenue-weighted demand", 0.0, 1.0, base.revenue, 0.05),
        severity=st.slider("Severity (negative sentiment)", 0.0, 1.0, base.severity, 0.05),
        momentum=st.slider("Momentum (significant growth)", 0.0, 1.0, base.momentum, 0.05),
    )

with st.sidebar.expander("Model"):
    semantic = get_language(language).default_embedding
    backend = st.selectbox(
        "Embedding backend",
        [semantic, "hybrid", "tfidf"],
        help=f"{semantic} = semantic sentence embeddings (default for this language); "
        "hybrid adds TF-IDF vocabulary; tfidf needs no model download.",
    )
    use_claude = st.toggle(
        "Review themes with Claude",
        value=False,
        disabled=not llm.available(),
        help="Names, classifies and de-duplicates themes. Needs ANTHROPIC_API_KEY.",
    )
    if not llm.available():
        st.caption("Set `ANTHROPIC_API_KEY` to enable the Claude analyst layer.")

# --------------------------------------------------------------------------- analysis
data_key = frame_key(feedback_raw) + backend + language + ",".join(product_names)
model = theme_model(data_key, feedback_raw, backend, language, product_names)
config = Config(language=language, embedding=backend, product_names=product_names, weights=weights)
if use_claude:
    try:
        first = analyze(feedback_raw, accounts_raw, releases_raw, config=config, model=model)
        model = claude_reviewed_model(data_key, model, first.themes)
    except llm.LLMUnavailable as exc:
        st.sidebar.warning(f"Claude unavailable: {exc}")
analysis = analyze(
    feedback_raw, accounts_raw, releases_raw, config=config, as_of=pd.Timestamp(as_of), model=model
)
themes = analysis.themes
names = dict(zip(themes["theme_id"], themes["name"], strict=True))
road = analysis.roadmap

if model.backend != backend:
    st.warning(
        f"The `{backend}` model could not be loaded, so Clamor fell back to "
        f"`{model.backend}`. Results will be less accurate."
    )

# --------------------------------------------------------------------------- header
st.title("What should we build next?")
fb = analysis.feedback[analysis.feedback["created_at"] < analysis.as_of + pd.Timedelta(days=1)]
cols = st.columns(5)
cols[0].metric("Feedback items", f"{len(fb):,}")
cols[1].metric(analysis.people.capitalize(), f"{fb['account_id'].nunique():,}")
cols[2].metric("Themes", len(themes))
cols[3].metric("Early warnings", int(themes["status"].isin(["new", "emerging"]).sum()))
if analysis.has_revenue:
    cols[4].metric("MRR represented", f"${fb.drop_duplicates('account_id')['mrr'].sum():,.0f}")
else:
    cols[4].metric("Revenue data", "not provided")

tab_road, tab_theme, tab_warn, tab_rel, tab_eval = st.tabs(
    ["Roadmap", "Theme explorer", "Early warning", "Release radar", "Model quality"]
)

# --------------------------------------------------------------------------- roadmap
with tab_road:
    insight_cols = st.columns(2)
    for i, ins in enumerate(headline_insights(analysis)):
        with insight_cols[i % 2].container(border=True):
            st.markdown(md(f"**{ins.title}**  \n{ins.detail}"))

    st.subheader("Priority ranking")
    w = weights.normalized()
    st.caption(
        f"Points each signal contributes. Weights: reach {w.reach:.0%}, revenue "
        f"{w.revenue:.0%}, severity {w.severity:.0%}, momentum {w.momentum:.0%}. "
        f"Reach, revenue and severity use the last {config.score_window_days} days."
    )
    parts = contributions(
        themes, weights if analysis.has_revenue else replace(weights, revenue=0.0)
    )
    st.plotly_chart(charts.priority_chart(themes, parts, mode=MODE), theme=None)
    st.dataframe(
        roadmap_table(analysis),
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.1f"
            )
        },
    )

    st.subheader("Counting votes vs. weighing evidence")
    st.caption(
        "Left: rank by raw number of mentions. Right: Clamor's priority. Highlighted "
        "themes moved the most."
    )
    st.plotly_chart(charts.rank_shift_chart(themes, mode=MODE), theme=None)

# --------------------------------------------------------------------------- theme explorer
with tab_theme:
    ranked = list(road["theme_id"])
    options = ranked + [t for t in themes["theme_id"] if t not in ranked]
    pick = st.selectbox("Theme", options, format_func=lambda t: f"{t} · {names[t]}")
    row = themes.set_index("theme_id").loc[pick]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### {row['name']}")
        if row.get("summary"):
            st.markdown(row["summary"])
        st.caption(
            f"{row['kind'].replace('_', ' ').capitalize()} · keywords: "
            f"{', '.join(row['keywords'][:6])}"
        )
        m = st.columns(4)
        m[0].metric(f"Mentions ({config.score_window_days}d)", int(row["mentions"]))
        m[1].metric(analysis.people.capitalize(), int(row["accounts"]))
        m[2].metric("Sentiment", f"{row['mean_sentiment']:+.2f}")
        m[3].metric("Trend", STATUS_ICON.get(row["status"], row["status"]))
        st.plotly_chart(
            charts.timeline_chart(
                analysis.weekly, names, [pick], analysis.releases, mode=MODE, normalize=False
            ),
            theme=None,
        )
    with right:
        st.markdown("**What customers say**")
        for q in list(row["examples"])[:6]:
            st.markdown(md(f"> {q}"))
        mix = pd.Series(row["plan_mix"], name="share").sort_values(ascending=False)
        if analysis.has_revenue and len(mix):
            st.markdown("**Plan mix**")
            st.dataframe(mix.map("{:.0%}".format))

    st.markdown("#### Opportunity brief")
    brief_llm = st.toggle(
        "Write it with Claude", value=False, disabled=not llm.available(), key="brief_llm"
    )
    text, src = write_brief(analysis, pick, use_llm=brief_llm)
    with st.container(border=True):
        st.markdown(md(text))
    st.download_button("Download brief (Markdown)", text, file_name=f"brief-{pick}.md")
    with st.expander("Evidence pack sent to the brief writer"):
        st.json(theme_evidence(analysis, pick), expanded=False)

# --------------------------------------------------------------------------- early warning
with tab_warn:
    st.caption(
        f"Each theme's mention rate in the last {config.recent_days} days vs. the "
        f"{config.baseline_days} days before, normalized for overall feedback volume "
        "(median-of-ratios), tested with an exact Poisson rate test and corrected for "
        "multiple comparisons (Benjamini-Hochberg). Tip: move **Analyze as of** in the "
        "sidebar to mid-March to watch the sync regression get flagged."
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
    warn["status"] = warn["status"].map(STATUS_ICON)
    st.dataframe(
        warn,
        hide_index=True,
        column_config={
            "theme_id": "ID",
            "name": "Theme",
            "status": "Status",
            "recent_mentions": "Recent",
            "baseline_mentions": "Baseline",
            "lift": st.column_config.NumberColumn("Rate vs baseline", format="x%.2f"),
            "lift_ci_low": st.column_config.NumberColumn("95% CI low", format="x%.2f"),
            "lift_ci_high": st.column_config.NumberColumn("95% CI high", format="x%.2f"),
            "q_value": st.column_config.NumberColumn("q-value (FDR)", format="%.4f"),
        },
    )
    default = [
        t for t in themes.loc[themes["status"].isin(["new", "emerging", "declining"]), "theme_id"]
    ][:4] or list(road["theme_id"][:3])
    chosen = st.multiselect(
        "Compare themes over time (up to 4)",
        options,
        default=default,
        max_selections=4,
        format_func=lambda t: f"{t} · {names[t]}",
    )
    st.plotly_chart(
        charts.timeline_chart(analysis.weekly, names, chosen, analysis.releases, mode=MODE),
        theme=None,
    )

# --------------------------------------------------------------------------- release radar
with tab_rel:
    if analysis.releases is None or analysis.releases.empty:
        st.info(
            "Upload a releases file (date, title, description) to see whether shipped "
            "work changed what customers talk about."
        )
    else:
        st.caption(
            "For each release, the linked theme's mention rate after vs. before "
            "(windows cut at neighbouring releases, normalized for overall volume). "
            "Observational evidence, not a controlled experiment."
        )
        st.plotly_chart(charts.release_chart(analysis.releases, names, mode=MODE), theme=None)
        st.dataframe(release_table(analysis), hide_index=True)
        if analysis.side_effects is not None and len(analysis.side_effects):
            st.markdown(
                "**Suspected side effects**: themes that spiked after a release they "
                "were not linked to."
            )
            fx = analysis.side_effects.assign(theme=analysis.side_effects["theme_id"].map(names))
            st.dataframe(
                fx[["version", "theme", "pre_mentions", "post_mentions", "rate_ratio", "q_value"]],
                hide_index=True,
            )

# --------------------------------------------------------------------------- model quality
with tab_eval:
    if dataset is None:
        st.info(
            "Model quality is measured on the synthetic dataset, where the true theme of "
            "every item is known. Switch to the demo data to see it."
        )
    elif pd.Timestamp(as_of) < fb_valid["created_at"].max().normalize():
        st.info("Move **Analyze as of** to the last day to run the evaluation.")
    else:
        ev = cached_evaluation(data_key, analysis, dataset)
        t, a = ev["themes"], ev["alerts"]
        c = st.columns(4)
        c[0].metric("Adjusted Rand index", f"{t['adjusted_rand']:.3f}")
        c[1].metric("Theme purity (homogeneity)", f"{t['homogeneity']:.3f}")
        c[2].metric("Sentiment sign accuracy", f"{ev['sentiment']['sign_accuracy']:.0%}")
        c[3].metric(
            "Releases linked correctly",
            f"{int(ev['releases']['correct'].sum())}/{len(ev['releases'])}",
        )
        st.markdown("**Early-warning backtest**: history replayed day by day.")
        summary = pd.DataFrame(a).T.rename(
            index={"clamor": "Clamor", "naive_2x": "Naive: week ≥ 2x average"}
        )
        st.dataframe(summary)
        st.dataframe(ev["detection"], hide_index=True)
        st.markdown("**How well each true theme was recovered**")
        st.dataframe(ev["per_theme"], hide_index=True)

st.caption(
    "Clamor · open source · synthetic demo data · [GitHub](https://github.com/ArdaSeyhan34/clamor)"
)
