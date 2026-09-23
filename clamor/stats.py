"""Small statistical helpers shared by trend detection and release impact analysis.

Both questions ("is this theme growing?" and "did this release change it?") compare how
often a theme is mentioned in two time windows, A and B.

**Normalization.** Raw counts mislead because overall volume moves too: a survey blast
inflates everything, and a single exploding bug inflates the total so much that every
*other* theme's share appears to drop. Clamor borrows *median-of-ratios* normalization from
RNA-seq differential expression (DESeq2): most themes do not change between two windows,
so the median of the per-theme ratios A/B is a robust estimate of how much the overall
volume changed. A theme is only interesting if it deviates from that.

**Test.** Given n = a + b mentions of a theme and the expected ratio f = E[a]/E[b] under
"no change", a ~ Binomial(n, f / (1 + f)). This is the exact conditional test for a ratio
of two Poisson rates and stays valid for small counts, which is exactly when early
warnings matter. The Clopper-Pearson interval on the binomial proportion converts into a
confidence interval for the rate ratio.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

SMOOTH = 0.5


@dataclass(frozen=True)
class RateComparison:
    count_a: int
    count_b: int
    rate_ratio: float  # observed a/b relative to the expected a/b; 1.0 = no change
    ci_low: float
    ci_high: float
    p_value: float


def volume_ratio(
    counts_a: Mapping[str, int],
    counts_b: Mapping[str, int],
    total_a: int,
    total_b: int,
    min_count: int = 10,
    min_themes: int = 4,
    method: str = "median_of_ratios",
) -> float:
    """Median-of-ratios estimate of E[a]/E[b] for a theme that did not change.

    Falls back to the ratio of total volumes when too few themes have enough data, or when
    ``method="total"`` (kept for the ablation study).
    """
    if method == "total":
        return total_a / total_b if total_a > 0 and total_b > 0 else float("nan")
    ratios = [
        (counts_a.get(k, 0) + SMOOTH) / (counts_b.get(k, 0) + SMOOTH)
        for k in set(counts_a) | set(counts_b)
        if counts_a.get(k, 0) + counts_b.get(k, 0) >= min_count
    ]
    if len(ratios) >= min_themes:
        return float(np.exp(np.median(np.log(ratios))))
    if total_a > 0 and total_b > 0:
        return total_a / total_b
    return float("nan")


def compare_rates(count_a: int, count_b: int, expected_ratio: float) -> RateComparison:
    """Is ``count_a / count_b`` different from `expected_ratio`?"""
    if not np.isfinite(expected_ratio) or expected_ratio <= 0:
        nan = float("nan")
        return RateComparison(count_a, count_b, nan, nan, nan, 1.0)
    ratio = ((count_a + SMOOTH) / (count_b + SMOOTH)) / expected_ratio
    n = count_a + count_b
    if n == 0:
        return RateComparison(count_a, count_b, ratio, 0.0, float("inf"), 1.0)
    p0 = expected_ratio / (1.0 + expected_ratio)
    test = binomtest(count_a, n, p0, alternative="two-sided")
    ci = test.proportion_ci(confidence_level=0.95, method="exact")

    def to_ratio(pi: float) -> float:
        return float("inf") if pi >= 1.0 else pi / (1.0 - pi) / expected_ratio

    return RateComparison(
        count_a, count_b, ratio, to_ratio(ci.low), to_ratio(ci.high), float(test.pvalue)
    )


def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
    """False-discovery-rate adjusted p-values (q-values)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def format_q(q: float) -> str:
    """Human-friendly q/p-value: '< 0.001' instead of '0.000'."""
    return "< 0.001" if q < 0.001 else f"{q:.3f}"
