"""Measuring Clamor against the simulator's ground truth.

Unsupervised pipelines are easy to demo and hard to trust. Because the synthetic dataset
records the true theme of every item, the true polarity, which theme each release was
meant to change and when each spike really started, every stage can be scored:

* **Theme discovery**: adjusted Rand index, NMI, homogeneity/completeness, and per-theme
  recovery (how much of each true theme lands in themes that mostly contain it).
* **Sentiment**: sign accuracy on items whose true polarity is negative or positive.
* **Release matching**: did the release notes get linked to the right theme?
* **Early warning backtest**: replaying history day by day, how many days after a spike
  started did Clamor raise an alert, and how many false alarms did it raise? Compared
  with the "this week is 2x the recent average" rule that many dashboards use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics

from .pipeline import Analysis
from .trends import detect_trends

ALERT_STATUSES = {"new", "emerging"}


def theme_mapping(analysis: Analysis, truth: pd.DataFrame) -> pd.Series:
    """Map each discovered theme to the true theme most of its items belong to."""
    fb = analysis.feedback.merge(truth, on="feedback_id")
    fb = fb[fb["theme_id"].notna()]
    return fb.groupby("theme_id")["true_theme"].agg(lambda s: s.value_counts().index[0])


def evaluate_themes(analysis: Analysis, truth: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    fb = analysis.feedback.merge(truth, on="feedback_id")
    pred = fb["theme_id"].fillna("(none)").astype(str)
    true = fb["true_theme"]
    summary = {
        "items": len(fb),
        "themes_found": int(pred[pred != "(none)"].nunique()),
        "true_themes": int(true.nunique()),
        "adjusted_rand": metrics.adjusted_rand_score(true, pred),
        "nmi": metrics.normalized_mutual_info_score(true, pred),
        "homogeneity": metrics.homogeneity_score(true, pred),
        "completeness": metrics.completeness_score(true, pred),
        "unassigned_share": float((pred == "(none)").mean()),
    }
    mapping = theme_mapping(analysis, truth)
    mapped = pred.map(mapping)
    rows = []
    for theme, group in fb.groupby("true_theme"):
        mine = mapped[group.index] == theme
        clusters = mapping[mapping == theme].index.tolist()
        counts = pred[group.index].value_counts()
        best = counts.index[0]
        rows.append(
            {
                "true_theme": theme,
                "items": len(group),
                "recovered_share": float(mine.mean()),  # lands in a theme mostly about it
                "themes": len(clusters),
                "largest_theme": best,
                "largest_theme_share": float(counts.iloc[0] / len(group)),
                "largest_theme_purity": float((true[pred == best] == theme).mean()),
            }
        )
    per_theme = pd.DataFrame(rows).sort_values("items", ascending=False)
    summary["mean_recovered_share"] = float(per_theme["recovered_share"].mean())
    return summary, per_theme


def evaluate_sentiment(analysis: Analysis, truth: pd.DataFrame) -> dict:
    fb = analysis.feedback.merge(truth, on="feedback_id")
    polar = fb[fb["true_polarity"].isin(["negative", "positive"])]
    predicted = np.where(polar["sentiment"] >= 0, "positive", "negative")
    by_class = polar.assign(pred=predicted).groupby("true_polarity")[["pred", "true_polarity"]]
    return {
        "items": len(polar),
        "sign_accuracy": float((predicted == polar["true_polarity"]).mean()),
        "recall_by_polarity": {
            k: float((g["pred"] == g["true_polarity"]).mean()) for k, g in by_class
        },
        "mean_sentiment_by_polarity": fb.groupby("true_polarity")["sentiment"].mean().to_dict(),
    }


def evaluate_release_matching(
    analysis: Analysis, truth: pd.DataFrame, release_truth: pd.DataFrame
) -> pd.DataFrame:
    if analysis.releases is None:
        return pd.DataFrame()
    mapping = theme_mapping(analysis, truth)
    out = analysis.releases[["version", "title", "theme_id", "verdict"]].merge(
        release_truth, on="version", how="left"
    )
    out["matched_true_theme"] = out["theme_id"].map(mapping)
    out["correct"] = out["matched_true_theme"] == out["intended_theme"]
    return out


def _naive_alerts(daily: pd.DataFrame, day: pd.Timestamp) -> set[str]:
    """Dashboard-style rule: last 7 days >= 2x the weekly average of the prior 28 days."""
    last7 = daily.loc[day - pd.Timedelta(days=6) : day].sum()
    prior = daily.loc[day - pd.Timedelta(days=34) : day - pd.Timedelta(days=7)].sum() / 4
    hit = (last7 >= 2 * prior) & (last7 >= 5)
    return set(hit[hit].index)


def backtest_alerts(
    analysis: Analysis,
    truth: pd.DataFrame,
    events: pd.DataFrame,
    warmup_days: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Replay every day of history and compare alerting rules against known spikes."""
    cfg = analysis.config
    mapping = theme_mapping(analysis, truth)
    fb = analysis.feedback
    mentions = analysis.model.doc_themes.merge(fb[["created_at"]], left_on="doc", right_index=True)
    dates = fb["created_at"]
    theme_ids = analysis.model.themes["theme_id"].tolist()
    day_index = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="D")
    daily = (
        mentions.assign(day=mentions["created_at"].dt.normalize())
        .groupby(["day", "theme_id"])["doc"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=day_index, columns=theme_ids, fill_value=0)
    )

    events = events.copy()
    events["start_date"] = pd.to_datetime(events["start_date"])
    spikes = events[events["kind"] == "spike"]
    expected = events[events["kind"].isin(["spike", "gradual growth"])]

    def is_expected(true_theme: str, day: pd.Timestamp) -> bool:
        rows = expected[expected["theme"] == true_theme]
        for _, e in rows.iterrows():
            end = pd.to_datetime(e["end_date"]) if pd.notna(e["end_date"]) else None
            if day >= e["start_date"] and (end is None or day <= end + pd.Timedelta(days=28)):
                return True
        return False

    log = []
    for day in day_index[warmup_days:]:
        trends = detect_trends(
            mentions,
            dates[dates < day + pd.Timedelta(days=1)],
            theme_ids,
            day,
            cfg.recent_days,
            cfg.baseline_days,
            cfg.alpha,
        )
        clamor = set(trends.loc[trends["status"].isin(ALERT_STATUSES), "theme_id"])
        for method, alerts in (("clamor", clamor), ("naive_2x", _naive_alerts(daily, day))):
            for tid in alerts:
                true_theme = mapping.get(tid)
                log.append(
                    {
                        "day": day,
                        "method": method,
                        "theme_id": tid,
                        "true_theme": true_theme,
                        "expected": bool(true_theme) and is_expected(true_theme, day),
                    }
                )
    log = pd.DataFrame(log, columns=["day", "method", "theme_id", "true_theme", "expected"])

    rows = []
    for _, spike in spikes.iterrows():
        for method in ("clamor", "naive_2x"):
            hits = log[
                (log["method"] == method)
                & (log["true_theme"] == spike["theme"])
                & (log["day"] >= spike["start_date"])
            ]
            first = hits["day"].min() if len(hits) else pd.NaT
            rows.append(
                {
                    "spike": spike["theme"],
                    "started": spike["start_date"].date(),
                    "method": method,
                    "first_alert": None if pd.isna(first) else first.date(),
                    "days_to_detect": None
                    if pd.isna(first)
                    else (first - spike["start_date"]).days,
                }
            )
    detection = pd.DataFrame(rows)

    summary = {}
    for method in ("clamor", "naive_2x"):
        false = log[(log["method"] == method) & ~log["expected"]].sort_values("day")
        # consecutive alert days for the same theme count as one false alarm episode
        episodes = 0
        for _, g in false.groupby("theme_id"):
            episodes += int((g["day"].diff() != pd.Timedelta(days=1)).sum())
        det = detection[detection["method"] == method]
        summary[method] = {
            "spikes_detected": int(det["days_to_detect"].notna().sum()),
            "spikes_total": int(len(det)),
            "median_days_to_detect": float(det["days_to_detect"].median()),
            "false_alarm_episodes": episodes,
            "false_alarm_days": int(len(false)),
        }
    return detection, summary


def evaluate_all(analysis: Analysis, dataset) -> dict:
    """Run every evaluation against a :class:`clamor.synth.SyntheticDataset`."""
    theme_summary, per_theme = evaluate_themes(analysis, dataset.ground_truth)
    detection, alert_summary = backtest_alerts(analysis, dataset.ground_truth, dataset.events)
    return {
        "themes": theme_summary,
        "per_theme": per_theme,
        "sentiment": evaluate_sentiment(analysis, dataset.ground_truth),
        "releases": evaluate_release_matching(
            analysis, dataset.ground_truth, dataset.release_truth
        ),
        "detection": detection,
        "alerts": alert_summary,
    }
