"""Probe tests. No snapshot needed."""

import numpy as np
import pandas as pd
import pytest

from src.erlang import required_agents
from src.probe import PRIMARY, compare_staffing, perturb, to_hourly

STAFFING_C = dict(PRIMARY, model="c", patience_seconds=None)


def test_perturb_hits_target_wape_and_is_reproducible():
    actual = np.random.default_rng(0).integers(1, 200, 5000).astype(float)
    a = perturb(actual, 0.10, np.random.default_rng(3))
    b = perturb(actual, 0.10, np.random.default_rng(3))
    assert np.abs(a - actual).sum() / actual.sum() == pytest.approx(0.10, abs=1e-9)
    assert np.array_equal(a, b)


def test_compare_staffing_identical_arrays_report_no_change():
    actual = np.array([5.0, 20.0, 80.0])
    got = compare_staffing(actual, actual.copy(), **PRIMARY)
    assert got["change_rate"] == 0.0
    assert got["mean_abs_diff"] == 0.0


def test_compare_staffing_known_difference():
    # Zero calls need zero agents; 0.001 calls need exactly one.
    got = compare_staffing(np.array([0.0, 0.0]), np.array([0.0, 0.001]), **PRIMARY)
    assert got == {
        "change_rate": 0.5,
        "within_one_rate": 1.0,
        "mean_abs_diff": 0.5,
        "mean_signed_diff": 0.5,
    }


@pytest.mark.parametrize("staffing", [PRIMARY, STAFFING_C])
def test_threshold_lookup_matches_required_agents(staffing):
    calls = np.array([0.0, 0.3, 4.7, 18.2, 33.3, 61.9, 112.4, 250.5])
    got = compare_staffing(np.zeros_like(calls), calls, **staffing)
    expected = np.mean(
        [required_agents(c, **staffing) for c in calls]
    )
    assert got["mean_signed_diff"] == pytest.approx(expected)
    # Per-value check: a mean can hide offsetting errors.
    for c in calls:
        single = compare_staffing(np.array([0.0]), np.array([c]), **staffing)
        assert single["mean_signed_diff"] == required_agents(c, **staffing)


def test_to_hourly_drops_hours_with_one_half():
    index = pd.to_datetime(["2025-01-06 09:00", "2025-01-06 09:30", "2025-01-06 10:00"])
    series = pd.DataFrame({"demand": [10, 12, 7]}, index=index)
    assert list(to_hourly(series)) == [22.0]
