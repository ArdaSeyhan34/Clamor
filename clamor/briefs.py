"""One-page opportunity briefs: the hand-off from analysis to a product decision.

A brief is generated from an *evidence pack*: every number and quote in it comes from the
analysis. With an API key, Claude writes the prose (see :mod:`clamor.llm`); without one, a
deterministic template produces the same structure, so the output is always usable.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import i18n, llm
from .i18n import t
from .pipeline import Analysis

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


def template_brief(ev: dict, lang: str = "en") -> str:
    window = next(k for k in ev if k.startswith("mentions_last_")).split("_")[2]
    mentions = ev[f"mentions_last_{window}_days"]
    accounts = ev[f"accounts_last_{window}_days"]
    word = ev.get("people", "accounts")
    people = i18n.people(word, lang)
    plans = ", ".join(
        f"{k} {i18n.pct(v, lang)}"
        for k, v in sorted(ev.get("plan_mix", {}).items(), key=lambda kv: -kv[1])[:3]
    )
    channels = ", ".join(
        f"{i18n.channel(k, lang)} {i18n.pct(v, lang)}"
        for k, v in sorted(ev["channel_mix"].items(), key=lambda kv: -kv[1])[:3]
    )
    trend = ev["trend"]
    trend_line = t(
        {
            "new": "It appeared recently and was absent before.",
            "emerging": "It is **growing fast**",
            "rising": "It is growing",
            "declining": "It is **declining**",
            "stable": "Its share of feedback is stable",
            "quiet": "There is too little recent data to call a trend",
        }.get(trend["status"], "Trend unknown"),
        lang,
    )
    if trend["status"] in {"emerging", "rising", "declining", "stable"}:
        lo, hi = trend["ratio_95ci"]
        trend_line += t(
            ": the rate over the last {days} days is {ratio}x the baseline "
            "(95% CI {lo}-{hi}, FDR q {q}).",
            lang,
            days=trend["recent_window_days"],
            ratio=i18n.number(trend["recent_vs_baseline_rate_ratio"], lang, 2),
            lo=i18n.number(lo, lang, 2),
            hi=i18n.number(hi, lang, 2),
            q=i18n.q_value(trend["fdr_q_value"], lang),
        )
    kind_key = next((k for k, v in KIND_LABEL.items() if v == ev["type"]), "ux")
    key = t("highest-revenue" if "arr_of_accounts_raising_it" in ev else "most frequent", lang)
    options = [
        t(o, lang).format(people=i18n.people(word, lang, "plural"), key=key)
        for o in OPTIONS_BY_KIND.get(kind_key, OPTIONS_BY_KIND["ux"])
    ]

    lines = [
        f"## {ev['name']}",
        "",
        f"*{t(ev['type'], lang)} · {t('theme {id}', lang, id=ev['theme_id'])}*",
        "",
        f"### {t('Problem', lang)}",
        t("Customers report: “{quote}.”", lang, quote=ev["representative_quote"]),
        "",
        f"### {t('Who is affected', lang)}",
        t(
            "- **{mentions}** mentions from **{n}** {people} in the last {window} days "
            "({total} all time; {share} of all feedback in that window)",
            lang,
            mentions=mentions,
            n=accounts,
            people=people,
            window=window,
            total=ev["mentions_all_time"],
            share=i18n.pct(ev["share_of_all_feedback"], lang, 1),
        ),
        *([t("- Plans: {plans}", lang, plans=plans)] if plans else []),
        t("- Channels: {channels}", lang, channels=channels),
    ]
    if "arr_of_accounts_raising_it" in ev:
        lines.append(
            t(
                "- Accounts raising it represent **{arr} ARR**; revenue-weighted demand is "
                "{mrr} MRR",
                lang,
                arr=i18n.money(ev["arr_of_accounts_raising_it"], lang),
                mrr=i18n.money(ev["revenue_weighted_demand_mrr"], lang),
            )
        )
    lines += [
        "",
        f"### {t('Evidence', lang)}",
        *[f"- \u201c{q}\u201d" for q in ev["quotes"][:5]],
        "",
        f"### {t('Why now', lang)}",
        trend_line,
    ]
    if "priority" in ev:
        p = ev["priority"]
        lines.append(
            t(
                "\nPriority score **{score}/100**, rank #{rank} (#{votes} if we only counted "
                "mentions).",
                lang,
                score=i18n.number(p["score_0_100"], lang, 1),
                rank=p["rank"],
                votes=p["rank_by_raw_mention_count"],
            )
        )
    for r in ev.get("related_releases", []):
        ratio = r["rate_ratio_after_vs_before"]
        extra = (
            t(", mention rate x{ratio} afterwards", lang, ratio=i18n.number(ratio, lang, 2))
            if ratio is not None
            else ""
        )
        lines.append(
            t(
                "\n- Release **{version}** ({date}, {title}): *{verdict}*{extra}",
                lang,
                version=r["version"],
                date=r["date"],
                title=r["title"],
                verdict=t(r["verdict"], lang),
                extra=extra,
            )
        )
    lines += [
        "",
        f"### {t('Options to explore', lang)}",
        *[f"1. {o}" for o in options],
        "",
        f"### {t('How we will know it worked', lang)}",
        t(
            "- This theme's mention rate drops significantly in Clamor's release radar within "
            "4 weeks of shipping",
            lang,
        ),
        t(
            "- Sentiment of remaining mentions improves; no new theme spikes after the release",
            lang,
        ),
        "",
        f"### {t('Open questions', lang)}",
        t(
            "- Which of the quoted {people} can we talk to this week?",
            lang,
            people=i18n.people(word, lang, "plural"),
        ),
        t("- Is there usage data that confirms the size of the problem?", lang),
    ]
    return "\n".join(lines)


def write_brief(
    analysis: Analysis,
    theme_id: str,
    use_llm: bool | None = None,
    analyst: llm.ClaudeAnalyst | None = None,
    lang: str | None = None,
) -> tuple[str, str]:
    """Returns (markdown, source) where source is 'claude' or 'template'.

    ``lang`` is the language of the brief (default: the config's output language).
    """
    lang = lang or analysis.config.output_language
    evidence = theme_evidence(analysis, theme_id)
    if use_llm is None:
        use_llm = analyst is not None or llm.available()
    if use_llm:
        try:
            analyst = analyst or llm.ClaudeAnalyst()
            return analyst.write_brief(evidence, language=lang), "claude"
        except llm.LLMUnavailable as exc:
            log.warning("Claude brief unavailable (%s); using the template.", exc)
    return template_brief(evidence, lang), "template"
