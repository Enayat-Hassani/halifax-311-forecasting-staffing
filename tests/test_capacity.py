"""Capacity-horizon tests on synthetic weekly series. No snapshot needed."""

import numpy as np
import pandas as pd
import pytest

from src.capacity import FTE, TEST, SeasonalNaive, XGBoost, backtest, costs, weekly


def _weeks(n=560, start="2016-09-05"):
    index = pd.date_range(start, periods=n, freq="W-MON")
    return pd.DataFrame({"demand": np.arange(n, dtype=float) + 1000, "closed": 0}, index=index)


class _Spy:
    name = "spy"

    def __init__(self):
        self.seen = []

    def forecast(self, history, future):
        self.seen.append((history.index.max(), future.index.min(), future.columns.tolist()))
        return np.zeros(len(future))


def test_forecast_sees_only_weeks_before_the_origin():
    weeks, spy = _weeks(), _Spy()
    backtest(spy, weeks, 13, TEST[:5])
    for (last, first, columns), target in zip(spy.seen, TEST[:5]):
        assert first == target - pd.Timedelta(weeks=13)
        assert last == first - pd.Timedelta(weeks=1)
        assert columns == ["closed"]


def test_each_test_week_scored_once():
    got = backtest(SeasonalNaive(), _weeks(), 26, TEST)
    assert len(got) == 52 and got["week"].is_unique


def test_seasonal_naive_is_the_same_week_last_year():
    weeks = _weeks()
    got = backtest(SeasonalNaive(), weeks, 13, TEST[:3])
    expected = weeks["demand"].reindex(TEST[:3] - pd.Timedelta(weeks=52)).to_numpy()
    assert (got["forecast"].to_numpy() == expected).all()


def test_xgboost_lags_reach_no_closer_than_the_lead():
    weeks = _weeks()
    features = XGBoost.features(weeks["demand"], weeks, 14)
    lags = [int(c[4:-1]) for c in features if c.startswith("lag_")]
    assert min(lags) == 14
    with pytest.raises(ValueError):
        XGBoost.features(weeks["demand"], weeks, 0)


def test_fte_constants_do_not_change_relative_cost():
    rng = np.random.default_rng(0)
    actual = rng.uniform(3000, 5000, 52)
    test = pd.DataFrame({"actual": actual, "forecast": actual * rng.uniform(0.8, 1.2, 52)})
    calib = pd.DataFrame({"actual": actual, "forecast": actual * rng.uniform(0.8, 1.2, 52)})
    base = costs(test, calib)
    scaled = costs(test, calib, dict(FTE, occupancy=0.7, shrinkage=0.4, paid_hours=35.0))
    ratio = scaled / base
    assert np.allclose(ratio, ratio.iloc[0, 0])


def test_weekly_sums_open_intervals_and_counts_closed_days():
    index = pd.date_range("2017-07-03", periods=2 * 336, freq="30min")
    series = pd.DataFrame({"demand": 1.0, "is_open": True, "day_kind": "normal"}, index=index)
    series.loc["2017-07-11", ["is_open", "day_kind"]] = [False, "closed"]
    got = weekly(series)
    assert got["demand"].tolist() == [336.0, 288.0]
    assert got["closed"].tolist() == [0, 1]
