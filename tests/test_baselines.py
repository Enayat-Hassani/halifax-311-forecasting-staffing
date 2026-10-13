"""Baseline model tests on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import (
    WEEK,
    MSTLModel,
    ProphetModel,
    SeasonalNaive,
    XGBoostModel,
    lag_features,
)


def _history(weeks=4):
    index = pd.date_range("2025-01-06", periods=weeks * WEEK, freq="30min")
    rng = np.random.default_rng(0)
    shape = np.where((index.hour >= 8) & (index.hour < 20), 30.0, 0.0)
    return pd.Series(shape + rng.poisson(3, len(index)), index=index)


def test_seasonal_naive_repeats_the_last_week():
    history = _history()
    got = SeasonalNaive().forecast(history, 2 * WEEK)
    last = history.to_numpy()[-WEEK:]
    assert np.array_equal(got, np.concatenate([last, last]))


@pytest.mark.parametrize("model", [MSTLModel(), ProphetModel(yearly=False), XGBoostModel()])
def test_fitted_models_return_finite_horizon(model):
    got = model.forecast(_history(8), 2 * WEEK)
    assert got.shape == (2 * WEEK,)
    assert np.isfinite(got).all()


def test_lag_features_read_nothing_within_14_days_of_the_slot():
    y = _history(8)
    slot = y.index[-1]
    before = lag_features(y, pd.DatetimeIndex([slot]))
    changed = y.copy()
    changed[changed.index > slot - pd.Timedelta(days=14)] = -1.0
    pd.testing.assert_frame_equal(lag_features(changed, pd.DatetimeIndex([slot])), before)
    # The boundary itself is read: the 14-day lag is the value 672 slots earlier.
    assert before.loc[slot, "lag_2w"] == y.iloc[-1 - 2 * WEEK]


def test_lag_features_values_and_missing_yearly_lag():
    index = pd.date_range("2025-01-06", periods=8 * WEEK, freq="30min")
    y = pd.Series(np.arange(len(index), dtype=float), index=index)
    got = lag_features(y, index[-1:]).iloc[0]
    n = len(index) - 1
    assert [got[f"lag_{k}w"] for k in (2, 3, 4, 5)] == [n - k * WEEK for k in (2, 3, 4, 5)]
    assert got["lag_mean"] == n - 3.5 * WEEK
    assert np.isnan(got["lag_52w"])
    assert (got["slot"], got["dow"]) == (47, 6)


def test_xgboost_rejects_a_horizon_past_the_shortest_lag():
    with pytest.raises(ValueError):
        XGBoostModel().forecast(_history(8), 2 * WEEK + 1)


def test_xgboost_is_deterministic():
    history = _history(8)
    assert np.array_equal(
        XGBoostModel().forecast(history, WEEK), XGBoostModel().forecast(history, WEEK)
    )
