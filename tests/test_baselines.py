"""Baseline model tests on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import WEEK, MSTLModel, ProphetModel, SeasonalNaive


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


@pytest.mark.parametrize("model", [MSTLModel(), ProphetModel(yearly=False)])
def test_fitted_models_return_finite_horizon(model):
    got = model.forecast(_history(), 2 * WEEK)
    assert got.shape == (2 * WEEK,)
    assert np.isfinite(got).all()
