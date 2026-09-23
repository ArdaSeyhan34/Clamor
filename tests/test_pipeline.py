import pandas as pd
import pytest

from clamor.config import PRESETS
from clamor.pipeline import analyze, merge_themes, rerun, rescore
from clamor.trends import STATUS_ORDER

VERDICTS = {
    "Resolved",
    "Improved",
    "No detectable change",
    "Worse",
    "Inconclusive",
    "No matching theme",
    "Not enough data",
}


def test_analysis_structure(analysis):
    themes = analysis.themes
    assert len(themes) >= 8
    assert themes["theme_id"].is_unique
    assert set(themes["status"]) <= set(STATUS_ORDER)
    road = analysis.roadmap
    assert road["score"].is_monotonic_decreasing
    assert road["rank"].tolist() == list(range(1, len(road) + 1))
    # every item with content gets a primary theme, and mentions reference real themes
    assert analysis.feedback["theme_id"].notna().mean() > 0.95
    assert set(analysis.mentions["theme_id"]) <= set(themes["theme_id"])


def test_release_radar_has_verdicts(analysis, dataset):
    rel = analysis.releases
    shipped = pd.to_datetime(dataset.releases["date"]) <= analysis.as_of  # no future releases
    assert len(rel) == shipped.sum() >= 4
    assert set(rel["verdict"]) <= VERDICTS
    assert rel["theme_id"].notna().all()


def test_as_of_hides_the_future(analysis):
    earlier = rerun(analysis, as_of="2026-03-01")
    assert earlier.releases["date"].max() <= pd.Timestamp("2026-03-01")
    assert earlier.themes["mentions_all_time"].sum() < analysis.themes["mentions_all_time"].sum()


def test_rescore_reorders_without_recomputing(analysis):
    enterprise = rescore(analysis, PRESETS["enterprise"])
    growth = rescore(analysis, PRESETS["growth"])
    assert set(enterprise["theme_id"]) == set(growth["theme_id"])
    top_rev = enterprise.dropna(subset=["score"]).iloc[0]
    assert top_rev["c_revenue"] >= growth.dropna(subset=["score"]).iloc[0]["c_revenue"] - 1e-9


def test_merge_themes_keeps_ids_and_moves_mentions(analysis):
    road = analysis.roadmap
    a, b = road["theme_id"].iloc[0], road["theme_id"].iloc[1]
    merged = merge_themes(analysis.model, {b: a})
    assert b not in set(merged.themes["theme_id"]) and a in set(merged.themes["theme_id"])
    after = rerun(analysis, model=merged).themes.set_index("theme_id")
    before = analysis.themes.set_index("theme_id")
    assert after.loc[a, "mentions_all_time"] >= before.loc[a, "mentions_all_time"]


def test_feedback_only_input_works():
    fb = pd.DataFrame(
        {
            "comment": [
                "The app crashes on login",
                "Please add dark mode",
                "App crashes daily",
                "Dark mode would be great",
                "Login crash again",
                "Need a dark theme",
            ]
            * 4,
            "date": pd.date_range("2026-01-01", periods=24, freq="D"),
        }
    )
    from clamor.config import Config

    result = analyze(fb, config=Config(embedding="tfidf", min_theme_share=0.0))
    assert not result.has_revenue
    assert result.releases is None
    assert len(result.themes) >= 1


def test_missing_columns_raise_a_clear_error():
    from clamor.io import SchemaError, load_feedback

    with pytest.raises(SchemaError, match="text"):
        load_feedback(pd.DataFrame({"date": ["2026-01-01"]}))
