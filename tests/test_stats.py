import numpy as np
import pytest

from clamor.stats import benjamini_hochberg, compare_rates, volume_ratio


def test_no_change_is_not_significant():
    cmp = compare_rates(50, 50, expected_ratio=1.0)
    assert cmp.rate_ratio == pytest.approx(1.0)
    assert cmp.p_value > 0.5
    assert cmp.ci_low < 1 < cmp.ci_high


def test_large_change_is_significant_and_ci_covers_estimate():
    cmp = compare_rates(90, 30, expected_ratio=1.0)
    assert cmp.p_value < 1e-6
    assert cmp.ci_low < cmp.rate_ratio < cmp.ci_high
    assert cmp.ci_low > 1.5


def test_expected_ratio_accounts_for_volume():
    # window A has twice the traffic, so twice the mentions is "no change"
    cmp = compare_rates(100, 50, expected_ratio=2.0)
    assert cmp.p_value > 0.5 and cmp.rate_ratio == pytest.approx(1.0, rel=0.02)


def test_median_of_ratios_ignores_a_single_exploding_theme():
    before = {f"t{i}": 40 for i in range(8)}
    after = {f"t{i}": 40 for i in range(8)}
    after["t0"] = 400  # one theme explodes and inflates the total
    total_ratio = sum(after.values()) / sum(before.values())
    assert total_ratio > 2
    assert volume_ratio(after, before, sum(after.values()), sum(before.values())) == (
        pytest.approx(1.0)
    )


def test_benjamini_hochberg_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.2])
    np.testing.assert_allclose(benjamini_hochberg(p), [0.04, 0.0533333, 0.0533333, 0.2], rtol=1e-5)
