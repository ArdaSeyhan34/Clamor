"""Plain-language headlines: what a PM should take away from an analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .pipeline import Analysis
from .stats import format_q


@dataclass(frozen=True)
class Insight:
    kind: str  # priority | loud | valuable | alert | release | regression
    title: str
    detail: str


def _name(row: pd.Series) -> str:
    return str(row.get("name") or row["label"])


def _money(x: float) -> str:
    if x >= 1_000_000:
        return f"${x / 1_000_000:,.1f}M"
    return f"${x / 1000:,.1f}k" if x >= 1000 else f"${x:,.0f}"


def _top_plan(mix: dict) -> tuple[str, float]:
    if not isinstance(mix, dict) or not mix:
        return "unknown", 0.0
    plan = max(mix, key=mix.get)
    return plan, mix[plan]


def headline_insights(analysis: Analysis, max_items: int = 6) -> list[Insight]:
    road = analysis.roadmap
    out: list[Insight] = []
    if road.empty:
        return out
    window = analysis.config.score_window_days

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
            f"Top priority: {_name(top)}",
            f"Score {top['score']:.0f}/100, driven mostly by {main[0]} and {main[1]}: "
            f"{int(top['mentions'])} mentions from {int(top['accounts'])} accounts in the last "
            f"{window} days.",
        )
    )

    alerts = road[road["status"].isin(["new", "emerging"])].sort_values("lift", ascending=False)
    for _, r in alerts.head(2).iterrows():
        out.append(
            Insight(
                "alert",
                f"Early warning: {_name(r)} is {r['status']}",
                f"Mention rate is {r['lift']:.1f}x its baseline over the last "
                f"{analysis.config.recent_days} days (95% CI {r['lift_ci_low']:.1f}-"
                f"{r['lift_ci_high']:.1f}, q {format_q(r['q_value'])}).",
            )
        )

    loud = road[road["rank_shift"] < 0].sort_values("rank_shift").head(1)
    for _, r in loud.iterrows():
        plan, share = _top_plan(r["plan_mix"])
        extra = (
            f" {share:.0%} of its mentions come from {plan} accounts."
            if analysis.has_revenue
            else ""
        )
        out.append(
            Insight(
                "loud",
                f"Loud, but not the most valuable: {_name(r)}",
                f"#{int(r['vote_rank'])} by raw mention count, "
                f"#{int(r['rank'])} by priority.{extra}",
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
                    f"Quiet, but expensive: {_name(r)}",
                    f"Only #{int(r['vote_rank'])} by mention count, yet the largest "
                    f"revenue-weighted demand ({_money(r['mrr_weighted'])} MRR; "
                    f"accounts raising it hold "
                    f"{_money(r['arr_exposed'])} ARR). {share:.0%} of mentions come from {plan}.",
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
            theme = names.get(r["theme_id"], r["theme_id"])
            change = f"{(r['rate_ratio'] - 1):+.0%}"
            out.append(
                Insight(
                    "release",
                    f"{r['version']} ({r['title']}): {r['verdict']}",
                    f"Mentions of “{theme}” changed {change} relative to overall feedback "
                    f"after the release (95% CI x{r['ci_low']:.2f}-x{r['ci_high']:.2f}).",
                )
            )
    return out[:max_items]
