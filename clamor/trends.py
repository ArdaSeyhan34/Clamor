"""Early warning: which themes are growing faster than feedback overall?

For an ``as_of`` date, each theme's mention rate in the *recent* window (default: last 28
days) is compared with a *baseline* window (the 84 days before that) using the exact
conditional rate test in :mod:`clamor.stats`. Because many themes are tested at once,
p-values are corrected with Benjamini-Hochberg so the alert list controls its false
discovery rate instead of crying wolf.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import benjamini_hochberg, compare_rates, volume_ratio

STATUS_ORDER = ["new", "emerging", "rising", "stable", "declining", "quiet"]


def window_counts(
    mentions: pd.DataFrame, dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Series, int]:
    """Theme mention counts and total feedback volume in [start, end)."""
    in_window = (mentions["created_at"] >= start) & (mentions["created_at"] < end)
    counts = mentions.loc[in_window].groupby("theme_id")["doc"].nunique()
    total = int(((dates >= start) & (dates < end)).sum())
    return counts, total


def detect_trends(
    mentions: pd.DataFrame,
    dates: pd.Series,
    theme_ids: list[str],
    as_of: pd.Timestamp,
    recent_days: int = 28,
    baseline_days: int = 84,
    alpha: float = 0.05,
    min_lift: float = 1.5,
    normalization: str = "median_of_ratios",
) -> pd.DataFrame:
    """Classify every theme as new / emerging / rising / stable / declining / quiet.

    `mentions` has one row per (doc, theme_id) with the item's ``created_at``;
    `dates` holds the timestamp of every feedback item (for total volume).
    """
    end = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
    recent_start = end - pd.Timedelta(days=recent_days)
    base_start = max(recent_start - pd.Timedelta(days=baseline_days), dates.min().normalize())
    recent, n_recent = window_counts(mentions, dates, recent_start, end)
    base, n_base = window_counts(mentions, dates, base_start, recent_start)

    expected = volume_ratio(
        recent.to_dict(), base.to_dict(), n_recent, n_base, method=normalization
    )
    rows = []
    for tid in theme_ids:
        a, b = int(recent.get(tid, 0)), int(base.get(tid, 0))
        cmp = compare_rates(a, b, expected)
        rows.append(
            {
                "theme_id": tid,
                "recent_mentions": a,
                "baseline_mentions": b,
                "recent_share": a / n_recent if n_recent else np.nan,
                "baseline_share": b / n_base if n_base else np.nan,
                "lift": cmp.rate_ratio,
                "lift_ci_low": cmp.ci_low,
                "lift_ci_high": cmp.ci_high,
                "p_value": cmp.p_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out.assign(q_value=[], status=[])
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())

    def status(r: pd.Series) -> str:
        if r.recent_mentions + r.baseline_mentions < 5:
            return "quiet"
        significant = r.q_value < alpha
        if significant and r.baseline_mentions == 0:
            return "new"
        if significant and r.lift >= min_lift:
            return "emerging"
        if significant and r.lift > 1:
            return "rising"
        if significant and r.lift < 1:
            return "declining"
        return "stable"

    out["status"] = out.apply(status, axis=1)
    out.attrs.update(
        as_of=pd.Timestamp(as_of),
        recent_start=recent_start,
        baseline_start=base_start,
        recent_total=n_recent,
        baseline_total=n_base,
        volume_ratio=expected,
    )
    return out


def weekly_mentions(mentions: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Weekly mention counts per theme (columns) plus total feedback volume."""
    week = mentions["created_at"].dt.to_period("W-SUN").dt.start_time
    table = (
        mentions.assign(week=week)
        .groupby(["week", "theme_id"])["doc"]
        .nunique()
        .unstack(fill_value=0)
    )
    totals = dates.dt.to_period("W-SUN").dt.start_time.value_counts().sort_index()
    table = table.reindex(totals.index, fill_value=0)
    table["__total__"] = totals.to_numpy()
    table.index.name = "week"
    return table
