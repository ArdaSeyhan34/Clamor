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
from clamor.io import load_accounts, load_feedback, load_releases  # noqa: E402
from clamor.pipeline import analyze, build_theme_model, merge_themes  # noqa: E402
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
DEMO_DIR = ROOT / "data" / "demo"


# --------------------------------------------------------------------------- data & models
@st.cache_data(show_spinner=False)
def load_demo() -> dict:
    if not (DEMO_DIR / "feedback.csv").exists():
        synth.generate().save(DEMO_DIR)
    ds = synth.SyntheticDataset.load(DEMO_DIR)
    return {name: getattr(ds, name) for name in ds.FILES}


@st.cache_resource(show_spinner=False)
def embedder_for(backend: str):
    return get_embedder(backend)


@st.cache_resource(show_spinner="Discovering themes (embedding and clustering)...")
def theme_model(data_key: str, _feedback: pd.DataFrame, backend: str, product: tuple[str, ...]):
    fb = load_feedback(_feedback)
    cfg = Config(embedding=backend, product_names=product)
    return build_theme_model(fb, cfg, embedder=embedder_for(backend), sentiment=score_feedback(fb))


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

source = st.sidebar.radio(
    "Data", ["Demo: Tempo (synthetic SaaS)", "Upload your own"], label_visibility="collapsed"
)
dataset = None
if source.startswith("Demo"):
    dataset = load_demo()
    feedback_raw, accounts_raw, releases_raw = (
        dataset["feedback"],
        dataset["accounts"],
        dataset["releases"],
    )
    product_names: tuple[str, ...] = ("Tempo",)
else:
    up_fb = st.sidebar.file_uploader("Feedback CSV (text + date required)", type="csv")
    up_acc = st.sidebar.file_uploader("Accounts CSV (account_id + mrr, optional)", type="csv")
    up_rel = st.sidebar.file_uploader("Releases CSV (date + title, optional)", type="csv")
    names = st.sidebar.text_input("Product name(s) to ignore, comma separated", "")
    product_names = tuple(n.strip() for n in names.split(",") if n.strip())
    if up_fb is None:
        st.title("Bring your own feedback")
        st.markdown(
            "Upload a CSV export from your support tool, app store reviews or NPS survey. "
            "Only a **text** column and a **date** column are required; common names such as "
            "`body`, `comment`, `review`, `created`, `timestamp` are recognized automatically."
            "\n\nAdd an **accounts** file (`account_id`, `mrr`, `plan`) to weigh themes by "
            "revenue, and a **releases** file (`date`, `title`, `description`) to get the "
            "release radar. Nothing leaves your browser session except, if you enable it, "
            "theme summaries sent to Claude."
        )
        st.stop()
    feedback_raw = pd.read_csv(up_fb)
    accounts_raw = pd.read_csv(up_acc) if up_acc else None
    releases_raw = pd.read_csv(up_rel) if up_rel else None

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
    backend = st.selectbox(
        "Embedding backend",
        ["minilm", "hybrid", "tfidf"],
        help="minilm = semantic sentence embeddings (default); hybrid adds TF-IDF "
        "vocabulary; tfidf needs no model download.",
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
data_key = frame_key(feedback_raw) + backend + ",".join(product_names)
model = theme_model(data_key, feedback_raw, backend, product_names)
config = Config(embedding=backend, product_names=product_names, weights=weights)
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
cols[1].metric("Accounts", f"{fb['account_id'].nunique():,}")
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
        m[1].metric("Accounts", int(row["accounts"]))
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
    "Clamor · open source · synthetic demo data · [GitHub](https://github.com/ArdaSeyhan34/NewRepo)"
)
