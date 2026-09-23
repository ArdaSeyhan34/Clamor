import pandas as pd
import pytest

from clamor.config import PRESETS, Weights
from clamor.scoring import contributions, score_themes


def _themes():
    return pd.DataFrame(
        {
            "theme_id": ["loud", "valuable", "growing", "praise"],
            "kind": ["feature_request", "feature_request", "bug", "praise"],
            "accounts": [300, 20, 40, 200],
            "mentions": [500, 30, 60, 300],
            "mrr_weighted": [2_000, 40_000, 5_000, 10_000],
            "negative_share": [0.1, 0.2, 0.8, 0.0],
            "lift": [1.0, 1.0, 3.0, 1.0],
            "status": ["stable", "stable", "emerging", "stable"],
        }
    )


def test_praise_is_not_on_the_roadmap():
    scored = score_themes(_themes(), Weights())
    assert scored.set_index("theme_id").loc["praise", "score"] != scored["score"].max()
    assert pd.isna(scored.set_index("theme_id").loc["praise", "score"])


def test_presets_change_the_winner():
    growth = score_themes(_themes(), PRESETS["growth"]).iloc[0]["theme_id"]
    enterprise = score_themes(_themes(), PRESETS["enterprise"]).iloc[0]["theme_id"]
    assert growth == "loud"
    assert enterprise == "valuable"


def test_contributions_sum_to_score_and_rank_shift_sign():
    w = Weights()
    scored = score_themes(_themes(), w)
    parts = contributions(scored, w)
    road = scored["score"].notna()
    assert parts[road].sum(axis=1).to_numpy() == pytest.approx(scored.loc[road, "score"])
    loud = scored.set_index("theme_id").loc["loud"]
    assert loud["vote_rank"] == 1  # most mentions
    assert loud["rank_shift"] <= 0  # never ranked higher by Clamor than by votes


def test_momentum_only_counts_when_significant():
    t = _themes()
    t.loc[2, "status"] = "stable"  # same lift, but not significant
    scored = score_themes(t, Weights()).set_index("theme_id")
    assert scored.loc["growing", "c_momentum"] == 0
