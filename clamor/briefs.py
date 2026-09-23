"""One-page opportunity briefs: the hand-off from analysis to a product decision.

A brief is generated from an *evidence pack*: every number and quote in it comes from the
analysis. With an API key, Claude writes the prose (see :mod:`clamor.llm`); without one, a
deterministic template produces the same structure, so the output is always usable.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import llm
from .pipeline import Analysis
from .stats import format_q

log = logging.getLogger(__name__)

KIND_LABEL = {
    "bug": "Bug / reliability",
    "feature_request": "Feature request",
    "ux": "Usability",
    "pricing": "Pricing & packaging",
    "praise": "Praise",
    "other": "Other",
}

OPTIONS_BY_KIND = {
    "bug": [
        "Reproduce with the {people} behind these quotes and add monitoring on the failing path",
        "Ship a targeted fix behind a flag and watch this theme's mention rate afterwards",
        "Proactively tell affected {people} what happened and when it will be fixed",
    ],
    "feature_request": [
        "Interview 5 of the requesting {people} to find the job behind the request",
        "Scope the smallest version that unblocks the {key} requesters",
        "Check whether an integration or workaround covers most of the need today",
    ],
    "ux": [
        "Run a quick usability test on the flow named in the quotes",
        "Add in-product guidance or better defaults before redesigning the flow",
        "Instrument the flow to measure where users drop off",
    ],
    "pricing": [
        "Segment the complaints by plan and seat count to see who is price sensitive",
        "Test packaging changes (e.g. inactive-seat billing) before list-price changes",
        "Arm customer success with a value narrative for renewal conversations",
    ],
}


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def theme_evidence(analysis: Analysis, theme_id: str) -> dict:
    """Everything known about one theme, as plain data (safe to serialize or send to an LLM)."""
    row = analysis.themes.set_index("theme_id").loc[theme_id]
    window = analysis.config.score_window_days
    evidence = {
        "theme_id": theme_id,
        "name": row.get("name", row["label"]),
        "type": KIND_LABEL.get(row["kind"], row["kind"]),
        "representative_quote": row["headline"],
        "keywords": list(row["keywords"]),
        f"mentions_last_{window}_days": int(row["mentions"]),
        "mentions_all_time": int(row["mentions_all_time"]),
        f"accounts_last_{window}_days": int(row["accounts"]),
        "share_of_all_feedback": round(float(row["share"]), 4),
        "mean_sentiment_-1_to_1": round(float(row["mean_sentiment"]), 2),
        "negative_share": round(float(row["negative_share"]), 2),
        "people": analysis.people,  # "accounts", or "users" for a consumer app
        "channel_mix": row["channel_mix"],
        "trend": {
            "status": row["status"],
            "recent_vs_baseline_rate_ratio": round(float(row["lift"]), 2),
            "ratio_95ci": [
                round(float(row["lift_ci_low"]), 2),
                round(float(row["lift_ci_high"]), 2),
            ],
            "fdr_q_value": round(float(row["q_value"]), 4),
            "recent_window_days": analysis.config.recent_days,
        },
        "quotes": list(row["examples"])[:8],
    }
    if analysis.has_revenue:
        evidence["plan_mix"] = row["plan_mix"]
        evidence["arr_of_accounts_raising_it"] = round(float(row["arr_exposed"]))
        evidence["revenue_weighted_demand_mrr"] = round(float(row["mrr_weighted"]))
    if pd.notna(row.get("score")):
        evidence["priority"] = {
            "score_0_100": round(float(row["score"]), 1),
            "rank": int(row["rank"]),
            "rank_by_raw_mention_count": int(row["vote_rank"]),
        }
    if analysis.releases is not None and len(analysis.releases):
        rel = analysis.releases[analysis.releases["theme_id"] == theme_id]
        evidence["related_releases"] = [
            {
                "version": r["version"],
                "date": str(pd.Timestamp(r["date"]).date()),
                "title": r["title"],
                "verdict": r["verdict"],
                "rate_ratio_after_vs_before": None
                if pd.isna(r.get("rate_ratio"))
                else round(float(r["rate_ratio"]), 2),
            }
            for _, r in rel.iterrows()
        ]
    return evidence


def template_brief(ev: dict) -> str:
    window = next(k for k in ev if k.startswith("mentions_last_")).split("_")[2]
    mentions = ev[f"mentions_last_{window}_days"]
    accounts = ev[f"accounts_last_{window}_days"]
    people = ev.get("people", "accounts")
    plans = ", ".join(
        f"{k} {v:.0%}" for k, v in sorted(ev.get("plan_mix", {}).items(), key=lambda kv: -kv[1])[:3]
    )
    channels = ", ".join(
        f"{k.replace('_', ' ')} {v:.0%}"
        for k, v in sorted(ev["channel_mix"].items(), key=lambda kv: -kv[1])[:3]
    )
    trend = ev["trend"]
    trend_line = {
        "new": "It appeared recently and was absent before.",
        "emerging": "It is **growing fast**",
        "rising": "It is growing",
        "declining": "It is **declining**",
        "stable": "Its share of feedback is stable",
        "quiet": "There is too little recent data to call a trend",
    }.get(trend["status"], "Trend unknown")
    if trend["status"] in {"emerging", "rising", "declining", "stable"}:
        lo, hi = trend["ratio_95ci"]
        trend_line += (
            f": the rate over the last {trend['recent_window_days']} days is "
            f"{trend['recent_vs_baseline_rate_ratio']:.2f}x the baseline "
            f"(95% CI {lo:.2f}-{hi:.2f}, FDR q {format_q(trend['fdr_q_value'])})."
        )
    kind_key = next((k for k, v in KIND_LABEL.items() if v == ev["type"]), "ux")
    key = "highest-revenue" if "arr_of_accounts_raising_it" in ev else "most frequent"
    options = [
        o.format(people=people, key=key)
        for o in OPTIONS_BY_KIND.get(kind_key, OPTIONS_BY_KIND["ux"])
    ]

    lines = [
        f"## {ev['name']}",
        "",
        f"*{ev['type']} · theme {ev['theme_id']}*",
        "",
        "### Problem",
        f"Customers report: “{ev['representative_quote']}.”",
        "",
        "### Who is affected",
        f"- **{mentions}** mentions from **{accounts}** {people} in the last {window} days "
        f"({ev['mentions_all_time']} all time; {ev['share_of_all_feedback']:.1%} of all feedback "
        "in that window)",
        *([f"- Plans: {plans}"] if plans else []),
        f"- Channels: {channels}",
    ]
    if "arr_of_accounts_raising_it" in ev:
        lines.append(
            f"- Accounts raising it represent **{_fmt_money(ev['arr_of_accounts_raising_it'])} "
            f"ARR**; revenue-weighted demand is {_fmt_money(ev['revenue_weighted_demand_mrr'])}"
            " MRR"
        )
    lines += [
        "",
        "### Evidence",
        *[f"- \u201c{q}\u201d" for q in ev["quotes"][:5]],
        "",
        "### Why now",
        trend_line,
    ]
    if "priority" in ev:
        p = ev["priority"]
        lines.append(
            f"\nPriority score **{p['score_0_100']}/100**, rank #{p['rank']} "
            f"(#{p['rank_by_raw_mention_count']} if we only counted mentions)."
        )
    for r in ev.get("related_releases", []):
        ratio = r["rate_ratio_after_vs_before"]
        extra = f", mention rate x{ratio:.2f} afterwards" if ratio is not None else ""
        lines.append(
            f"\n- Release **{r['version']}** ({r['date']}, {r['title']}): *{r['verdict']}*{extra}"
        )
    lines += [
        "",
        "### Options to explore",
        *[f"1. {o}" for o in options],
        "",
        "### How we will know it worked",
        "- This theme's mention rate drops significantly in Clamor's release radar within "
        "4 weeks of shipping",
        "- Sentiment of remaining mentions improves; no new theme spikes after the release",
        "",
        "### Open questions",
        f"- Which of the quoted {people} can we talk to this week?",
        "- Is there usage data that confirms the size of the problem?",
    ]
    return "\n".join(lines)


def write_brief(
    analysis: Analysis,
    theme_id: str,
    use_llm: bool | None = None,
    analyst: llm.ClaudeAnalyst | None = None,
) -> tuple[str, str]:
    """Returns (markdown, source) where source is 'claude' or 'template'."""
    evidence = theme_evidence(analysis, theme_id)
    if use_llm is None:
        use_llm = analyst is not None or llm.available()
    if use_llm:
        try:
            analyst = analyst or llm.ClaudeAnalyst()
            return analyst.write_brief(evidence), "claude"
        except llm.LLMUnavailable as exc:
            log.warning("Claude brief unavailable (%s); using the template.", exc)
    return template_brief(evidence), "template"
