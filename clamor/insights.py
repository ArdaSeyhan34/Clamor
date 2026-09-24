"""Plain-language headlines: what a PM should take away from an analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import i18n
from .i18n import t
from .pipeline import Analysis


@dataclass(frozen=True)
class Insight:
    kind: str  # priority | loud | valuable | alert | release | regression
    title: str
    detail: str


def _name(row: pd.Series) -> str:
    return str(row.get("name") or row["label"])


def _top_plan(mix: dict) -> tuple[str, float]:
    if not isinstance(mix, dict) or not mix:
        return "unknown", 0.0
    plan = max(mix, key=mix.get)
    return plan, mix[plan]


def _change(x: float, lang: str) -> str:
    """+230% in English; '%230 arttı' / '%85 azaldı' in Turkish."""
    if lang == "tr":
        return f"{i18n.pct(abs(x), lang)} {'arttı' if x >= 0 else 'azaldı'}"
    return i18n.pct(x, lang, signed=True)


def headline_insights(
    analysis: Analysis, max_items: int = 6, lang: str | None = None
) -> list[Insight]:
    """Up to ``max_items`` headlines, in ``lang`` (default: the config's output language)."""
    lang = lang or analysis.config.output_language
    road = analysis.roadmap
    out: list[Insight] = []
    if road.empty:
        return out
    window = analysis.config.score_window_days
    people = i18n.people(analysis.people, lang)

    def num(x: float, digits: int = 1) -> str:
        return i18n.number(x, lang, digits)

    top = road.iloc[0]
    drivers = {
        "reach": top["c_reach"],
        "revenue": top["c_revenue"],
        "severity": top["c_severity"],
        "momentum": top["c_momentum"],
    }
    main = sorted(drivers, key=drivers.get, reverse=True)[:2]
    out.append(
        Insight(
            "priority",
            t("Top priority: {name}", lang, name=_name(top)),
            t(
                "Score {score}/100, driven mostly by {a} and {b}: {mentions} mentions from "
                "{n} {people} in the last {window} days.",
                lang,
                score=f"{top['score']:.0f}",
                a=t(main[0], lang),
                b=t(main[1], lang),
                mentions=int(top["mentions"]),
                n=int(top["accounts"]),
                people=people,
                window=window,
            ),
        )
    )

    alerts = road[road["status"].isin(["new", "emerging"])].sort_values("lift", ascending=False)
    for _, r in alerts.head(2).iterrows():
        out.append(
            Insight(
                "alert",
                t(
                    "Early warning: {name} is {status}",
                    lang,
                    name=_name(r),
                    status=t(r["status"], lang),
                ),
                t(
                    "Mention rate is {lift}x its baseline over the last {days} days "
                    "(95% CI {lo}-{hi}, q {q}).",
                    lang,
                    lift=num(r["lift"]),
                    days=analysis.config.recent_days,
                    lo=num(r["lift_ci_low"]),
                    hi=num(r["lift_ci_high"]),
                    q=i18n.q_value(r["q_value"], lang),
                ),
            )
        )

    loud = road[road["rank_shift"] < 0].sort_values("rank_shift").head(1)
    for _, r in loud.iterrows():
        if analysis.has_revenue:
            plan, share = _top_plan(r["plan_mix"])
            title = t("Loud, but not the most valuable: {name}", lang, name=_name(r))
            extra = t(
                " {share} of its mentions come from {plan} accounts.",
                lang,
                share=i18n.pct(share, lang),
                plan=plan,
            )
        else:
            title = t("Frequently mentioned, but lower priority: {name}", lang, name=_name(r))
            w = analysis.config.weights
            gaps = {
                "fewer distinct users": w.reach * (top["c_reach"] - r["c_reach"]),
                "milder sentiment": w.severity * (top["c_severity"] - r["c_severity"]),
                "no significant growth": w.momentum * (top["c_momentum"] - r["c_momentum"]),
            }
            extra = t(
                " It ranks lower mainly because of {reason}.",
                lang,
                reason=t(max(gaps, key=gaps.get), lang),
            )
        out.append(
            Insight(
                "loud",
                title,
                t(
                    "#{votes} by raw mention count, #{rank} by priority.{extra}",
                    lang,
                    votes=int(r["vote_rank"]),
                    rank=int(r["rank"]),
                    extra=extra,
                ),
            )
        )

    if analysis.has_revenue:
        valuable = road.sort_values("mrr_weighted", ascending=False).head(1)
        for _, r in valuable.iterrows():
            if r["vote_rank"] <= 3:
                continue
            plan, share = _top_plan(r["plan_mix"])
            out.append(
                Insight(
                    "valuable",
                    t("Quiet, but expensive: {name}", lang, name=_name(r)),
                    t(
                        "Only #{votes} by mention count, yet the largest revenue-weighted "
                        "demand ({mrr} MRR; accounts raising it hold {arr} ARR). {share} of "
                        "mentions come from {plan}.",
                        lang,
                        votes=int(r["vote_rank"]),
                        mrr=i18n.money(r["mrr_weighted"], lang, compact=True),
                        arr=i18n.money(r["arr_exposed"], lang, compact=True),
                        share=i18n.pct(share, lang),
                        plan=plan,
                    ),
                )
            )

    if analysis.releases is not None and len(analysis.releases):
        names = analysis.themes.set_index("theme_id").apply(_name, axis=1)
        rel = analysis.releases
        for _, r in (
            rel[rel["verdict"].isin(["Worse", "Resolved", "No detectable change"])]
            .iloc[::-1]
            .head(2)
            .iterrows()
        ):
            out.append(
                Insight(
                    "release",
                    f"{r['version']} ({r['title']}): {t(r['verdict'], lang)}",
                    t(
                        "Mentions of “{theme}” changed {change} relative to overall feedback "
                        "after the release (95% CI x{lo}-x{hi}).",
                        lang,
                        theme=names.get(r["theme_id"], r["theme_id"]),
                        change=_change(r["rate_ratio"] - 1, lang),
                        lo=num(r["ci_low"], 2),
                        hi=num(r["ci_high"], 2),
                    ),
                )
            )
    return out[:max_items]
