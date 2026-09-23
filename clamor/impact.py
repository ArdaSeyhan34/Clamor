"""Release radar: closing the loop between what we shipped and what customers say.

Most feedback tools stop at "here are your top themes". Clamor also asks, for every
release in the changelog:

1. **Which theme was it meant to address?** The release notes are embedded with the same
   model as the feedback and matched to the closest theme.
2. **Did that theme's mention rate change afterwards?** Rates in the window before and
   after the release are compared with an exact test (see :mod:`clamor.stats`), giving a
   verdict: *Resolved*, *Improved*, *No detectable change*, *Worse*, or *Inconclusive*
   when there is too little data to tell.
3. **Did anything else spike?** Every other theme is tested for a post-release jump, with
   an FDR correction, to surface suspected regressions.

Windows are cut at neighbouring releases so one release's effect is not credited to
another. This is observational evidence, not a controlled experiment, and the UI says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import benjamini_hochberg, compare_rates, volume_ratio
from .trends import window_counts


def match_releases(
    releases: pd.DataFrame,
    release_vectors: np.ndarray,
    themes: pd.DataFrame,
    centroids: np.ndarray,
    min_similarity: float,
) -> pd.DataFrame:
    """Attach the most similar theme to every release (or none if nothing is close)."""
    out = releases.copy()
    if len(themes) == 0 or len(releases) == 0:
        out["theme_id"], out["match_similarity"] = None, np.nan
        return out
    sims = release_vectors @ centroids.T
    best = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(best)), best]
    ids = themes["theme_id"].to_numpy()
    out["theme_id"] = [
        ids[b] if s >= min_similarity else None for b, s in zip(best, best_sim, strict=True)
    ]
    out["match_similarity"] = best_sim
    return out


def _verdict(cmp, alpha: float) -> str:
    if np.isnan(cmp.rate_ratio):
        return "Not enough data"
    if cmp.p_value >= alpha:
        # distinguish "we looked and nothing moved" from "we cannot tell yet"
        too_few = cmp.count_a + cmp.count_b < 20
        too_wide = cmp.ci_low <= 0 or cmp.ci_high / cmp.ci_low > 6
        return "Inconclusive" if too_few or too_wide else "No detectable change"
    if cmp.rate_ratio < 1:
        return "Resolved" if cmp.rate_ratio <= 0.5 else "Improved"
    return "Worse"


def _windows(date: pd.Timestamp, all_dates: list[pd.Timestamp], window_days: int):
    """Pre/post windows around `date`, truncated at the neighbouring releases."""
    window = pd.Timedelta(days=window_days)
    prev_dates = [d for d in all_dates if d < date]
    next_dates = [d for d in all_dates if d > date]
    pre_start = max([date - window] + prev_dates)
    post_end = min([date + window] + next_dates)
    return pre_start, date, date, post_end


def release_radar(
    matched: pd.DataFrame,
    mentions: pd.DataFrame,
    dates: pd.Series,
    theme_ids: list[str],
    window_days: int = 28,
    alpha: float = 0.05,
    side_effect_lift: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (one row per release, one row per suspected side effect)."""
    release_dates = [pd.Timestamp(d).normalize() for d in matched["date"]]
    data_start, data_end = dates.min().normalize(), dates.max().normalize() + pd.Timedelta(days=1)
    rows, effects = [], []
    for (_, rel), date in zip(matched.iterrows(), release_dates, strict=True):
        pre_start, pre_end, post_start, post_end = _windows(date, release_dates, window_days)
        pre_start, post_end = max(pre_start, data_start), min(post_end, data_end)
        pre, n_pre = window_counts(mentions, dates, pre_start, pre_end)
        post, n_post = window_counts(mentions, dates, post_start, post_end)
        base = {
            "date": date,
            "version": rel.get("version"),
            "title": rel.get("title"),
            "theme_id": rel.get("theme_id"),
            "match_similarity": rel.get("match_similarity"),
            "pre_days": (pre_end - pre_start).days,
            "post_days": (post_end - post_start).days,
        }
        if n_pre == 0 or n_post == 0:
            rows.append({**base, "verdict": "Not enough data"})
            continue

        expected = volume_ratio(post.to_dict(), pre.to_dict(), n_post, n_pre)
        tid = rel.get("theme_id")
        if tid:
            cmp = compare_rates(int(post.get(tid, 0)), int(pre.get(tid, 0)), expected)
            rows.append(
                {
                    **base,
                    "pre_mentions": cmp.count_b,
                    "post_mentions": cmp.count_a,
                    "rate_ratio": cmp.rate_ratio,
                    "ci_low": cmp.ci_low,
                    "ci_high": cmp.ci_high,
                    "p_value": cmp.p_value,
                    "verdict": _verdict(cmp, alpha),
                }
            )
        else:
            rows.append({**base, "verdict": "No matching theme"})

        tests = [
            (t, compare_rates(int(post.get(t, 0)), int(pre.get(t, 0)), expected))
            for t in theme_ids
            if t != tid
        ]
        if tests:
            q = benjamini_hochberg([c.p_value for _, c in tests])
            for (t, c), qv in zip(tests, q, strict=True):
                if qv < alpha and c.rate_ratio >= side_effect_lift and c.count_a >= 5:
                    effects.append(
                        {
                            "version": rel.get("version"),
                            "date": date,
                            "theme_id": t,
                            "pre_mentions": c.count_b,
                            "post_mentions": c.count_a,
                            "rate_ratio": c.rate_ratio,
                            "q_value": qv,
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(effects)
