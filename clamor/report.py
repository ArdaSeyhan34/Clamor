"""Shareable outputs: a Markdown report (renders on GitHub), a self-contained HTML report
with interactive charts, a CSV of the roadmap and one brief per top opportunity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from jinja2 import Environment

from . import charts, i18n
from .briefs import write_brief
from .i18n import t
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


def md_table(df: pd.DataFrame, lang: str = "en") -> str:
    """GitHub-flavored Markdown table (no extra dependency)."""

    def cell(v) -> str:
        if lang == "tr" and isinstance(v, float):
            v = str(v).replace(".", ",")
        return str(v).replace("|", "\\|").replace("\n", " ")

    head = "| " + " | ".join(map(cell, df.columns)) + " |"
    sep = (
        "|"
        + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns)
        + "|"
    )
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def roadmap_table(
    analysis: Analysis, top_n: int | None = None, lang: str | None = None
) -> pd.DataFrame:
    lang = lang or analysis.config.output_language
    road = analysis.roadmap if top_n is None else analysis.roadmap.head(top_n)
    table = pd.DataFrame(
        {
            t("Rank", lang): road["rank"].astype(int),
            t("Theme", lang): road["name"],
            t("Type", lang): road["kind"].map(KIND).map(lambda k: t(k, lang)),
            t("Score", lang): road["score"].round(1),
            t("Mentions", lang): road["mentions"].astype(int),
            i18n.people(analysis.people, lang, "title"): road["accounts"].astype(int),
            t("Trend", lang): road["status"].map(lambda s: status_label(s, lang)),
            t("Rank by mentions", lang): road["vote_rank"].astype(int),
        }
    )
    if analysis.has_revenue:
        table.insert(
            6,
            t("Revenue-weighted MRR", lang),
            road["mrr_weighted"].map(lambda x: i18n.money(x, lang)),
        )
    return table


def status_label(status: str, lang: str = "en") -> str:
    """Trend status with its icon, e.g. '▲ emerging' / '▲ hızla artıyor'."""
    return t(STATUS_ICON.get(status, status), lang)


def release_table(analysis: Analysis, lang: str | None = None) -> pd.DataFrame:
    lang = lang or analysis.config.output_language
    rel = analysis.releases
    if rel is None or rel.empty:
        return pd.DataFrame()
    names = analysis.themes.set_index("theme_id")["name"]
    return pd.DataFrame(
        {
            t("Date", lang): pd.to_datetime(rel["date"]).dt.date,
            t("Release", lang): rel["version"] + " · " + rel["title"],
            t("Linked theme", lang): rel["theme_id"].map(names).fillna("–"),
            t("Before → after", lang): [
                "–" if pd.isna(a) else f"{int(a)} → {int(b)}"
                for a, b in zip(
                    rel.get("pre_mentions", pd.Series(dtype=float)),
                    rel.get("post_mentions", pd.Series(dtype=float)),
                    strict=False,
                )
            ]
            if "pre_mentions" in rel
            else "–",
            t("Rate change", lang): rel["rate_ratio"].map(
                lambda r: "–" if pd.isna(r) else f"x{i18n.number(r, lang, 2)}"
            )
            if "rate_ratio" in rel
            else "–",
            t("Verdict", lang): rel["verdict"].map(lambda v: t(v, lang)),
        }
    )


def to_markdown(analysis: Analysis, evaluation: dict | None = None, lang: str | None = None) -> str:
    lang = lang or analysis.config.output_language
    fb = analysis.feedback
    window = analysis.config.score_window_days

    def num(x: float, digits: int = 2) -> str:
        return i18n.number(x, lang, digits)

    lines = [
        f"# {t('Clamor report', lang)}",
        "",
        t(
            "*{items} feedback items from {n} {people}, {start} to {end}. {themes} themes "
            "discovered with the `{backend}` embedding backend. Priority reflects the last "
            "{window} days.*",
            lang,
            items=i18n.number(len(fb), lang),
            n=i18n.number(fb["account_id"].nunique(), lang),
            people=i18n.people(analysis.people, lang),
            start=i18n.date(fb["created_at"].min(), lang),
            end=i18n.date(analysis.as_of, lang),
            themes=len(analysis.themes),
            backend=analysis.model.backend,
            window=window,
        ),
        "",
        f"## {t('Key insights', lang)}",
        "",
    ]
    for ins in headline_insights(analysis, lang=lang):
        lines.append(f"- **{ins.title}.** {ins.detail}")
    lines += [
        "",
        f"## {t('Prioritized roadmap', lang)}",
        "",
        md_table(roadmap_table(analysis, 15, lang=lang), lang),
        "",
    ]
    alerts = analysis.roadmap[analysis.roadmap["status"].isin(["new", "emerging", "rising"])]
    lines += [f"## {t('Early warnings', lang)}", ""]
    if alerts.empty:
        lines.append(t("No theme is growing significantly faster than feedback overall.", lang))
    else:
        head = ["Theme", "Status", "Rate vs baseline", "95% CI", "q-value"]
        lines.append("| " + " | ".join(t(h, lang) for h in head) + " |")
        lines.append("|---|---|---|---|---|")
        for _, r in alerts.iterrows():
            lines.append(
                f"| {r['name']} | {status_label(r['status'], lang)} | x{num(r['lift'])} | "
                f"x{num(r['lift_ci_low'])} - x{num(r['lift_ci_high'])} | "
                f"{num(r['q_value'], 4)} |"
            )
    rel = release_table(analysis, lang=lang)
    if not rel.empty:
        lines += [
            "",
            f"## {t('Release radar', lang)}",
            "",
            t(
                "Did each release change what customers talk about? Rates are compared in "
                "windows of up to {days} days before and after each release, normalized for "
                "overall feedback volume.",
                lang,
                days=analysis.config.impact_window_days,
            ),
            "",
            md_table(rel, lang),
        ]
        if analysis.side_effects is not None and len(analysis.side_effects):
            names = analysis.themes.set_index("theme_id")["name"]
            lines += [
                "",
                t(
                    "**Suspected side effects** (themes that spiked after a release they "
                    "were not linked to):",
                    lang,
                ),
                "",
            ]
            for _, s in analysis.side_effects.iterrows():
                lines.append(
                    t(
                        "- {version}: *{theme}* x{ratio} ({pre} → {post} mentions, q = {q})",
                        lang,
                        version=s["version"],
                        theme=names.get(s["theme_id"], s["theme_id"]),
                        ratio=num(s["rate_ratio"], 1),
                        pre=int(s["pre_mentions"]),
                        post=int(s["post_mentions"]),
                        q=num(s["q_value"], 3),
                    )
                )
    if evaluation:
        te = evaluation["themes"]
        a = evaluation["alerts"]
        c, n = a["clamor"], a["naive_2x"]
        rows = [
            ("Adjusted Rand index", num(te["adjusted_rand"], 3)),
            ("Normalized mutual information", num(te["nmi"], 3)),
            ("Homogeneity (theme purity)", num(te["homogeneity"], 3)),
            (
                "Items landing in a theme about their true topic",
                i18n.pct(te["mean_recovered_share"], lang, 1),
            ),
            (
                "Sentiment sign accuracy",
                i18n.pct(evaluation["sentiment"]["sign_accuracy"], lang, 1),
            ),
            (
                "Releases linked to the right theme",
                f"{int(evaluation['releases']['correct'].sum())}/{len(evaluation['releases'])}",
            ),
            (
                "Spikes detected (Clamor / naive 2x rule)",
                f"{c['spikes_detected']}/{c['spikes_total']} / "
                f"{n['spikes_detected']}/{n['spikes_total']}",
            ),
            (
                "Median days to detect (Clamor / naive)",
                f"{c['median_days_to_detect']:.0f} / {n['median_days_to_detect']:.0f}",
            ),
            (
                "False alarm episodes (Clamor / naive)",
                f"{c['false_alarm_episodes']} / {n['false_alarm_episodes']}",
            ),
        ]
        lines += [
            "",
            f"## {t('Accuracy against ground truth (synthetic data)', lang)}",
            "",
            f"| {t('Metric', lang)} | {t('Value', lang)} |",
            "|---|---|",
            *[f"| {t(k, lang)} | {v} |" for k, v in rows],
        ]
    lines += [
        "",
        "---",
        t("*Generated by [Clamor](https://github.com/ArdaSeyhan34/clamor).*", lang),
        "",
    ]
    return "\n".join(lines)


HTML = """<!doctype html>
<html lang="{{ lang }}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ tx.title }}</title>
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
<h1>{{ tx.title }}</h1>
<p class="sub">{{ subtitle }}</p>
<div class="grid">
{% for k, v in tiles %}<div class="tile">
  <div class="v">{{ v }}</div><div class="k">{{ k }}</div></div>
{% endfor %}</div>

<h2>{{ tx.insights }}</h2>
<div class="insights">
{% for i in insights %}<div class="card insight">
  <b>{{ i.title }}</b><span>{{ i.detail }}</span></div>
{% endfor %}</div>

<h2>{{ tx.next }}</h2>
<p class="note">{{ tx.points }}</p>
<div class="card">{{ priority_chart }}</div>
<div class="scroll" style="margin-top:12px">{{ roadmap_html }}</div>

<h2>{{ tx.votes }}</h2>
<p class="note">{{ tx.votes_note }}</p>
<div class="card">{{ shift_chart }}</div>

<h2>{{ tx.timeline }}</h2>
<p class="note">{{ tx.timeline_note }}</p>
<div class="card">{{ timeline_chart }}</div>

{% if release_chart %}
<h2>{{ tx.radar }}</h2>
<p class="note">{{ tx.radar_note }}</p>
<div class="card">{{ release_chart }}</div>
<div class="scroll" style="margin-top:12px">{{ release_html }}</div>
{% endif %}

<footer>{{ tx.footer }}</footer>
</main></body></html>
"""


HTML_TEXT = {  # template slot -> English text (translated by clamor.i18n)
    "title": "Clamor report",
    "insights": "Key insights",
    "next": "What to work on next",
    "points": "Points each signal contributes to the priority score (weights: {weights}).",
    "votes": "Counting votes vs. weighing evidence",
    "votes_note": "Rank by raw number of mentions (left) vs. Clamor's priority (right). "
    "Highlighted lines moved the most.",
    "timeline": "How themes moved over time",
    "timeline_note": "Weekly share of all feedback. Vertical lines mark releases.",
    "radar": "Release radar",
    "radar_note": "Mention rate of the linked theme after vs. before each release, normalized "
    "for overall feedback volume, with 95% intervals. Observational evidence, not an "
    "experiment.",
    "footer": "Generated by Clamor · embedding backend {backend} · as of {date}",
}


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
    analysis: Analysis,
    timeline_themes: list[str] | None = None,
    offline: bool = False,
    lang: str | None = None,
) -> str:
    """Render the HTML report. `offline=True` inlines plotly.js (~4 MB) instead of the CDN."""
    lang = lang or analysis.config.output_language
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
        (t("feedback items", lang), i18n.number(len(fb), lang)),
        (
            i18n.people(analysis.people, lang, "plural"),
            i18n.number(fb["account_id"].nunique(), lang),
        ),
        (t("themes", lang), str(len(analysis.themes))),
        (
            t("early warnings", lang),
            str(int(analysis.themes["status"].isin(["new", "emerging"]).sum())),
        ),
    ]
    if analysis.has_revenue:
        tiles.append(
            (
                t("MRR represented", lang),
                i18n.money(fb.drop_duplicates("account_id")["mrr"].sum(), lang),
            )
        )
    rel_table = release_table(analysis, lang=lang)
    w = weights.normalized()
    weights_text = ", ".join(
        f"{t(k, lang)} {i18n.pct(getattr(w, k), lang)}"
        for k in ("reach", "revenue", "severity", "momentum")
    )
    backend, as_of = analysis.model.backend, f"{analysis.as_of:%Y-%m-%d}"
    env = Environment(autoescape=True)
    return env.from_string(HTML).render(
        lang=lang,
        tx={
            key: t(text, lang).format(weights=weights_text, backend=backend, date=as_of)
            for key, text in HTML_TEXT.items()
        },
        subtitle=t(
            "{items} feedback items · {start} to {end}",
            lang,
            items=i18n.number(len(fb), lang),
            start=i18n.date(fb["created_at"].min(), lang),
            end=i18n.date(analysis.as_of, lang),
        ),
        tiles=tiles,
        insights=headline_insights(analysis, lang=lang),
        priority_chart=_markup(embed(charts.priority_chart(analysis.themes, parts, lang=lang))),
        roadmap_html=_markup(roadmap_table(analysis, lang=lang).to_html(index=False, border=0)),
        shift_chart=_markup(embed(charts.rank_shift_chart(analysis.themes, lang=lang))),
        timeline_chart=_markup(
            embed(
                charts.timeline_chart(
                    analysis.weekly, names, timeline_themes, analysis.releases, lang=lang
                )
            )
        ),
        release_chart=_markup(embed(charts.release_chart(analysis.releases, names, lang=lang)))
        if analysis.releases is not None and len(analysis.releases)
        else None,
        release_html=_markup(rel_table.to_html(index=False, border=0)),
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
    lang: str | None = None,
) -> dict[str, Path]:
    """Write report.md, report.html, roadmap.csv and briefs/ in ``lang`` (default: the
    config's output language)."""
    lang = lang or analysis.config.output_language
    out = Path(out_dir)
    (out / "briefs").mkdir(parents=True, exist_ok=True)
    paths = {
        "markdown": out / "report.md",
        "html": out / "report.html",
        "roadmap": out / "roadmap.csv",
    }
    paths["markdown"].write_text(to_markdown(analysis, evaluation, lang=lang), encoding="utf-8")
    paths["html"].write_text(to_html(analysis, offline=offline, lang=lang), encoding="utf-8")
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
        text, _ = write_brief(analysis, row["theme_id"], use_llm=use_llm, lang=lang)
        path = out / "briefs" / f"{i:02d}-{row['theme_id']}.md"
        path.write_text(text + "\n", encoding="utf-8")
        paths[f"brief_{i}"] = path
    return paths
