"""Opportunity scoring: from theme statistics to a ranked, explainable roadmap.

Each actionable theme gets four components in [0, 1]:

reach     number of distinct accounts mentioning it
revenue   revenue-weighted demand: each account's MRR split across its mentions
severity  share of mentions with clearly negative sentiment
momentum  size of a statistically significant recent lift (0 if not significant)

Reach and revenue are scaled as sqrt(value / max): the square root dampens a single
whale account or viral thread without erasing real differences (a log scale compresses
them too much, a linear scale lets the top theme flatten everything else). The score is
the weighted sum, scaled to 0-100, and because the components are reported next to the
score, anyone can see *why* a theme ranks where it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Weights

NON_ACTIONABLE = {"praise"}
MOMENTUM_STATUSES = {"new", "emerging", "rising"}


def _sqrt_scale(values: pd.Series) -> pd.Series:
    values = values.clip(lower=0).astype(float)
    top = values.max()
    return np.sqrt(values / top) if top > 0 else values * 0.0


def score_themes(themes: pd.DataFrame, weights: Weights) -> pd.DataFrame:
    """Add component columns, `score`, `rank` and the naive `vote_rank` to a theme table."""
    w = weights.normalized()
    out = themes.copy()
    actionable = ~out["kind"].isin(NON_ACTIONABLE)
    out["c_reach"] = _sqrt_scale(out["accounts"].where(actionable, 0))
    out["c_revenue"] = _sqrt_scale(out["mrr_weighted"].where(actionable, 0))
    out["c_severity"] = out["negative_share"].clip(0, 1)
    lift = out.get("lift", pd.Series(1.0, index=out.index)).fillna(1.0)
    status = out.get("status", pd.Series("stable", index=out.index))
    out["c_momentum"] = np.where(
        status.isin(MOMENTUM_STATUSES), np.clip(np.log2(lift.clip(lower=1.0)) / 2.0, 0, 1), 0.0
    )
    out["score"] = 100 * (
        w.reach * out["c_reach"]
        + w.revenue * out["c_revenue"]
        + w.severity * out["c_severity"]
        + w.momentum * out["c_momentum"]
    )
    out.loc[~actionable, "score"] = np.nan
    out["rank"] = out["score"].rank(ascending=False, method="first")
    out["vote_rank"] = out["mentions"].where(actionable).rank(ascending=False, method="first")
    out["rank_shift"] = out["vote_rank"] - out["rank"]  # positive = Clamor ranks it higher
    return out.sort_values(["rank", "mentions"], ascending=[True, False], na_position="last")


def contributions(scored: pd.DataFrame, weights: Weights) -> pd.DataFrame:
    """Per-theme points contributed by each component (sums to the score)."""
    w = weights.normalized()
    parts = pd.DataFrame(
        {
            "Reach": 100 * w.reach * scored["c_reach"],
            "Revenue": 100 * w.revenue * scored["c_revenue"],
            "Severity": 100 * w.severity * scored["c_severity"],
            "Momentum": 100 * w.momentum * scored["c_momentum"],
        },
        index=scored.index,
    )
    return parts.where(scored["score"].notna())
