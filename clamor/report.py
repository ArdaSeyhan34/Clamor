"""Shareable outputs: a Markdown report (renders on GitHub), a self-contained HTML report
with interactive charts, a CSV of the roadmap and one brief per top opportunity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment

from . import charts
from .briefs import write_brief
from .insights import headline_insights
from .pipeline import Analysis
from .scoring import contributions

KIND = {
    "bug": "Bug",
    "feature_request": "Feature request",
    "ux": "Usability",
    "pricing": "Pricing",
    "praise": "Praise",
    "other": "Other",
}
STATUS_ICON = {
    "new": "★ new",
    "emerging": "▲ emerging",
    "rising": "↗ rising",
    "stable": "– stable",
    "declining": "↘ declining",
    "quiet": "· quiet",
}


def _money(x: float) -> str:
    return f"${x:,.0f}"


def md_table(df: pd.DataFrame) -> str:
    """GitHub-flavored Markdown table (no extra dependency)."""

    def cell(v) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(map(cell, df.columns)) + " |"
    sep = (
        "|"
        + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns)
        + "|"
    )
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def roadmap_table(analysis: Analysis, top_n: int | None = None) -> pd.DataFrame:
    road = analysis.roadmap if top_n is None else analysis.roadmap.head(top_n)
    table = pd.DataFrame(
        {
            "Rank": road["rank"].astype(int),
            "Theme": road["name"],
            "Type": road["kind"].map(KIND),
            "Score": road["score"].round(1),
            "Mentions": road["mentions"].astype(int),
            analysis.people.capitalize(): road["accounts"].astype(int),
            "Trend": road["status"].map(STATUS_ICON),
            "Rank by mentions": road["vote_rank"].astype(int),
        }
    )
    if analysis.has_revenue:
        table.insert(6, "Revenue-weighted MRR", road["mrr_weighted"].map(_money))
    return table


def release_table(analysis: Analysis) -> pd.DataFrame:
    rel = analysis.releases
    if rel is None or rel.empty:
        return pd.DataFrame()
    names = analysis.themes.set_index("theme_id")["name"]
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(rel["date"]).dt.date,
            "Release": rel["version"] + " · " + rel["title"],
            "Linked theme": rel["theme_id"].map(names).fillna("–"),
            "Before → after": [
                "–" if pd.isna(a) else f"{int(a)} → {int(b)}"
                for a, b in zip(
                    rel.get("pre_mentions", pd.Series(dtype=float)),
                    rel.get("post_mentions", pd.Series(dtype=float)),
                    strict=False,
                )
            ]
            if "pre_mentions" in rel
            else "–",
            "Rate change": rel["rate_ratio"].map(lambda r: "–" if pd.isna(r) else f"x{r:.2f}")
            if "rate_ratio" in rel
            else "–",
            "Verdict": rel["verdict"],
        }
    )


def to_markdown(analysis: Analysis, evaluation: dict | None = None) -> str:
    fb = analysis.feedback
    window = analysis.config.score_window_days
    lines = [
        "# Clamor report",
        "",
        f"*{len(fb):,} feedback items from {fb['account_id'].nunique():,} {analysis.people}, "
        f"{fb['created_at'].min():%b %d, %Y} to {analysis.as_of:%b %d, %Y}. "
        f"{len(analysis.themes)} themes discovered with the `{analysis.model.backend}` "
        f"embedding backend. Priority reflects the last {window} days.*",
        "",
        "## Key insights",
        "",
    ]
    for ins in headline_insights(analysis):
        lines.append(f"- **{ins.title}.** {ins.detail}")
    lines += ["", "## Prioritized roadmap", "", md_table(roadmap_table(analysis, 15)), ""]
    alerts = analysis.roadmap[analysis.roadmap["status"].isin(["new", "emerging", "rising"])]
    lines += ["## Early warnings", ""]
    if alerts.empty:
        lines.append("No theme is growing significantly faster than feedback overall.")
    else:
        lines.append("| Theme | Status | Rate vs baseline | 95% CI | q-value |")
        lines.append("|---|---|---|---|---|")
        for _, r in alerts.iterrows():
            lines.append(
                f"| {r['name']} | {STATUS_ICON[r['status']]} | x{r['lift']:.2f} | "
                f"x{r['lift_ci_low']:.2f} - x{r['lift_ci_high']:.2f} | "
                f"{r['q_value']:.4f} |"
            )
    rel = release_table(analysis)
    if not rel.empty:
        lines += [
            "",
            "## Release radar",
            "",
            "Did each release change what customers talk about? Rates are compared in "
            f"windows of up to {analysis.config.impact_window_days} days before and after "
            "each release, normalized for overall feedback volume.",
            "",
            md_table(rel),
        ]
        if analysis.side_effects is not None and len(analysis.side_effects):
            names = analysis.themes.set_index("theme_id")["name"]
            lines += [
                "",
                "**Suspected side effects** (themes that spiked after a release they "
                "were not linked to):",
                "",
            ]
            for _, s in analysis.side_effects.iterrows():
                lines.append(
                    f"- {s['version']}: *{names.get(s['theme_id'], s['theme_id'])}* "
                    f"x{s['rate_ratio']:.1f} ({int(s['pre_mentions'])} → "
                    f"{int(s['post_mentions'])} mentions, q = {s['q_value']:.3f})"
                )
    if evaluation:
        t = evaluation["themes"]
        a = evaluation["alerts"]
        lines += [
            "",
            "## Accuracy against ground truth (synthetic data)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Adjusted Rand index | {t['adjusted_rand']:.3f} |",
            f"| Normalized mutual information | {t['nmi']:.3f} |",
            f"| Homogeneity (theme purity) | {t['homogeneity']:.3f} |",
            f"| Items landing in a theme about their true topic | "
            f"{t['mean_recovered_share']:.1%} |",
            f"| Sentiment sign accuracy | {evaluation['sentiment']['sign_accuracy']:.1%} |",
            f"| Releases linked to the right theme | "
            f"{int(evaluation['releases']['correct'].sum())}/{len(evaluation['releases'])} |",
            f"| Spikes detected (Clamor / naive 2x rule) | {a['clamor']['spikes_detected']}/"
            f"{a['clamor']['spikes_total']} / {a['naive_2x']['spikes_detected']}/"
            f"{a['naive_2x']['spikes_total']} |",
            f"| Median days to detect (Clamor / naive) | {a['clamor']['median_days_to_detect']:.0f}"
            f" / {a['naive_2x']['median_days_to_detect']:.0f} |",
            f"| False alarm episodes (Clamor / naive) | {a['clamor']['false_alarm_episodes']} / "
            f"{a['naive_2x']['false_alarm_episodes']} |",
        ]
    lines += ["", "---", "*Generated by [Clamor](https://github.com/ArdaSeyhan34/clamor).*", ""]
    return "\n".join(lines)


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clamor report</title>
{{ plotly_js }}
<style>
  :root { color-scheme:light; --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b;
          --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --good:#006300; --bad:#d03b3b;
          --accent:#2a78d6; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:1080px; margin:0 auto; padding:32px 16px 64px; }
  h1 { font-size:28px; margin:0 0 4px; } h2 { font-size:19px; margin:40px 0 12px; }
  .sub { color:var(--ink2); margin:0 0 24px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }
  .tile, .card { background:var(--surface); border:1px solid var(--line); border-radius:10px;
                 padding:14px 16px; }
  .tile .v { font-size:24px; font-weight:600; } .tile .k { color:var(--ink2); font-size:13px; }
  .insights { display:grid; gap:10px; }
  .insight b { display:block; } .insight span { color:var(--ink2); }
  table { width:100%; border-collapse:collapse; font-size:14px; background:var(--surface); }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }
  th { color:var(--ink2); font-weight:600; } td.num, th.num { text-align:right;
       font-variant-numeric:tabular-nums; }
  .scroll { overflow-x:auto; border:1px solid var(--line); border-radius:10px; }
  .note { color:var(--muted); font-size:13px; }
  footer { margin-top:48px; color:var(--muted); font-size:13px; }
</style>
</head>
<body><main>
<h1>Clamor report</h1>
<p class="sub">{{ subtitle }}</p>
<div class="grid">
{% for k, v in tiles %}<div class="tile">
  <div class="v">{{ v }}</div><div class="k">{{ k }}</div></div>
{% endfor %}</div>

<h2>Key insights</h2>
<div class="insights">
{% for i in insights %}<div class="card insight">
  <b>{{ i.title }}</b><span>{{ i.detail }}</span></div>
{% endfor %}</div>

<h2>What to work on next</h2>
<p class="note">Points each signal contributes to the priority score (weights: {{ weights }}).</p>
<div class="card">{{ priority_chart }}</div>
<div class="scroll" style="margin-top:12px">{{ roadmap_html }}</div>

<h2>Counting votes vs. weighing evidence</h2>
<p class="note">Rank by raw number of mentions (left) vs. Clamor's priority (right).
Highlighted lines moved the most.</p>
<div class="card">{{ shift_chart }}</div>

<h2>How themes moved over time</h2>
<p class="note">Weekly share of all feedback. Vertical lines mark releases.</p>
<div class="card">{{ timeline_chart }}</div>

{% if release_chart %}
<h2>Release radar</h2>
<p class="note">Mention rate of the linked theme after vs. before each release, normalized
for overall feedback volume, with 95% intervals. Observational evidence, not an experiment.</p>
<div class="card">{{ release_chart }}</div>
<div class="scroll" style="margin-top:12px">{{ release_html }}</div>
{% endif %}

<footer>Generated by Clamor · embedding backend <code>{{ backend }}</code> ·
as of {{ as_of }}</footer>
</main></body></html>
"""


def default_timeline_themes(analysis: Analysis, n: int = 4) -> list[str]:
    """Themes with a story: alerts first, then themes a release clearly moved, then the top."""
    road = analysis.roadmap
    picks = list(road.loc[road["status"].isin(["new", "emerging"]), "theme_id"][:2])
    if analysis.releases is not None and len(analysis.releases):
        moved = analysis.releases[analysis.releases["verdict"].isin(["Worse", "Resolved"])]
        picks += [t for t in moved["theme_id"].dropna() if t not in picks]
    picks += [t for t in road["theme_id"] if t not in picks]
    return picks[:n]


PLOTLY_CDN = (
    '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
)


def to_html(
    analysis: Analysis, timeline_themes: list[str] | None = None, offline: bool = False
) -> str:
    """Render the HTML report. `offline=True` inlines plotly.js (~4 MB) instead of the CDN."""
    fb = analysis.feedback
    weights = analysis.config.weights
    parts = contributions(analysis.themes, weights)
    names = dict(zip(analysis.themes["theme_id"], analysis.themes["name"], strict=True))

    def embed(fig) -> str:
        return fig.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": False, "responsive": True},
        )

    if timeline_themes is None:
        timeline_themes = default_timeline_themes(analysis)
    tiles = [
        ("feedback items", f"{len(fb):,}"),
        (analysis.people, f"{fb['account_id'].nunique():,}"),
        ("themes", str(len(analysis.themes))),
        ("early warnings", str(int(analysis.themes["status"].isin(["new", "emerging"]).sum()))),
    ]
    if analysis.has_revenue:
        tiles.append(("MRR represented", _money(fb.drop_duplicates("account_id")["mrr"].sum())))
    rel_table = release_table(analysis)
    w = weights.normalized()
    env = Environment(autoescape=True)
    return env.from_string(HTML).render(
        subtitle=f"{len(fb):,} feedback items · {fb['created_at'].min():%b %d, %Y} to "
        f"{analysis.as_of:%b %d, %Y}",
        tiles=tiles,
        insights=headline_insights(analysis),
        weights=f"reach {w.reach:.0%}, revenue {w.revenue:.0%}, severity {w.severity:.0%}, "
        f"momentum {w.momentum:.0%}",
        priority_chart=_markup(embed(charts.priority_chart(analysis.themes, parts))),
        roadmap_html=_markup(roadmap_table(analysis).to_html(index=False, border=0)),
        shift_chart=_markup(embed(charts.rank_shift_chart(analysis.themes))),
        timeline_chart=_markup(
            embed(charts.timeline_chart(analysis.weekly, names, timeline_themes, analysis.releases))
        ),
        release_chart=_markup(embed(charts.release_chart(analysis.releases, names)))
        if analysis.releases is not None and len(analysis.releases)
        else None,
        release_html=_markup(rel_table.to_html(index=False, border=0)),
        backend=analysis.model.backend,
        as_of=f"{analysis.as_of:%Y-%m-%d}",
        plotly_js=_markup(_plotly_inline() if offline else PLOTLY_CDN),
    )


def _plotly_inline() -> str:
    from plotly.offline import get_plotlyjs

    return f"<script>{get_plotlyjs()}</script>"


def _markup(html: str):
    from markupsafe import Markup

    return Markup(html)


def write_report(
    analysis: Analysis,
    out_dir: str | Path,
    evaluation: dict | None = None,
    n_briefs: int = 3,
    use_llm: bool | None = None,
    offline: bool = False,
) -> dict[str, Path]:
    out = Path(out_dir)
    (out / "briefs").mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out / "report.md",
        "html": out / "report.html",
        "roadmap": out / "roadmap.csv",
    }
    paths["markdown"].write_text(to_markdown(analysis, evaluation), encoding="utf-8")
    paths["html"].write_text(to_html(analysis, offline=offline), encoding="utf-8")
    cols = [
        "rank",
        "theme_id",
        "name",
        "kind",
        "score",
        "c_reach",
        "c_revenue",
        "c_severity",
        "c_momentum",
        "mentions",
        "mentions_all_time",
        "accounts",
        "mrr_weighted",
        "arr_exposed",
        "mean_sentiment",
        "negative_share",
        "status",
        "lift",
        "q_value",
        "vote_rank",
        "headline",
    ]
    analysis.themes[[c for c in cols if c in analysis.themes]].to_csv(paths["roadmap"], index=False)
    for i, (_, row) in enumerate(analysis.roadmap.head(n_briefs).iterrows(), start=1):
        text, _ = write_brief(analysis, row["theme_id"], use_llm=use_llm)
        path = out / "briefs" / f"{i:02d}-{row['theme_id']}.md"
        path.write_text(text + "\n", encoding="utf-8")
        paths[f"brief_{i}"] = path
    return paths
