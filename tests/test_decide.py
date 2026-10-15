"""Decision-layer tests. No snapshot needed."""

import numpy as np
import pandas as pd
import pytest

from src.clean import WINDOWS
from src.decide import (
    CALIBRATION,
    PRIMARY,
    block_bootstrap,
    implied_service_level,
    poisson_floor,
    ratio_quantiles,
    staffing_cost,
)
from src.erlang import requirement_lookup
from src.evaluate import LEAD, TARGET


def test_ratio_quantiles_recover_a_constant_ratio():
    ts = pd.date_range("2025-03-03", periods=7 * 48, freq="30min")
    forecast = np.random.default_rng(0).uniform(5, 50, len(ts))
    calib = pd.DataFrame(
        {"ts": ts, "forecast": forecast, "actual": 1.2 * forecast, "is_open": True}
    )
    peak = {(False, s) for s in range(20, 30)}
    got = ratio_quantiles(calib, peak, np.array([0.2, 0.5, 0.9]))
    assert np.allclose(got.to_numpy(), 1.2)


def test_calibration_targets_end_inside_training():
    assert len(CALIBRATION) == 26
    assert (CALIBRATION + LEAD + TARGET).max() <= WINDOWS["primary"][1] + pd.Timedelta(days=1)


def test_staffing_cost_weights_short_by_the_ratio():
    assert staffing_cost([5, 5], [4, 7], 3.0) == pytest.approx(2.5)


def test_actual_requirement_meets_the_target():
    actual = np.array([3.0, 12.0, 40.0, 75.0, 140.0])
    agents = requirement_lookup(**PRIMARY, interval_seconds=1800.0)(actual)
    assert implied_service_level(actual, agents) >= PRIMARY["target"]
    assert implied_service_level(actual, agents - 1) < PRIMARY["target"]


def test_block_bootstrap_brackets_the_mean():
    assert block_bootstrap(np.full(52, 3.0)) == (3.0, 3.0)
    lo, hi = block_bootstrap(np.r_[np.zeros(26), np.ones(26)])
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_poisson_floor_matches_the_poisson_absolute_deviation():
    # For Poisson(100), E|X - 100| / 100 is close to sqrt(2 / (pi * 100)) = 0.0798.
    test = pd.DataFrame({"actual": np.full(5000, 100.0), "is_open": True})
    assert poisson_floor(test)["wape"] == pytest.approx(0.0798, abs=0.003)
