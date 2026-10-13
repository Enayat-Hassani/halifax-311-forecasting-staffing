"""Harness and metric tests on synthetic series. No snapshot needed."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import WEEK
from src.evaluate import (
    backtest,
    cleanse,
    origins,
    peak_keys,
    post_closure_days,
    score,
)


def _series(start="2017-07-03", end="2026-09-06", kinds=None):
    """Regular series whose demand encodes the slot's day number, so copies are traceable."""
    index = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=30), freq="30min")
    day = (index.normalize() - index[0]).days.to_numpy()
    series = pd.DataFrame(
        {"demand": day * 100.0 + index.hour, "is_open": True, "day_kind": "normal"}, index=index
    )
    for date, kind in (kinds or {}).items():
        series.loc[date, "day_kind"] = kind
        if kind == "closed":
            series.loc[date, ["demand", "is_open"]] = [0.0, False]
    return series


def test_cleanse_copies_previous_same_weekday_and_chains():
    s = _series("2024-01-01", "2024-02-11", {"2024-01-22": "closed", "2024-01-29": "closed",
                                             "2024-02-07": "outage"})
    got = cleanse(s)
    assert (got["2024-01-22"].to_numpy() == s.loc["2024-01-15", "demand"].to_numpy()).all()
    assert (got["2024-01-29"].to_numpy() == s.loc["2024-01-15", "demand"].to_numpy()).all()
    assert (got["2024-02-07"].to_numpy() == s.loc["2024-01-31", "demand"].to_numpy()).all()


def test_cleanse_leaves_open_days_and_target_unchanged():
    s = _series("2024-01-01", "2024-01-28", {"2024-01-22": "closed"})
    before = s["demand"].copy()
    got = cleanse(s)
    normal = s["day_kind"] == "normal"
    assert (got[normal] == s.loc[normal, "demand"]).all()
    pd.testing.assert_series_equal(s["demand"], before)


def test_origins_cover_52_weeks_of_the_test_window():
    got = origins(_series())
    assert len(got) == 52
    assert (got.dayofweek == 0).all()
    assert got[0] + pd.Timedelta(days=7) == pd.Timestamp("2025-09-01")
    assert got[-1] + pd.Timedelta(days=13) == pd.Timestamp("2026-08-30")


class Recorder:
    name = "recorder"

    def __init__(self):
        self.seen = []

    def forecast(self, history, horizon):
        self.seen.append((history.index[0], history.index[-1]))
        return np.full(horizon, -1.0)


@pytest.mark.parametrize("train_window, start", [("primary", "2024-07-17"),
                                                 ("secondary", "2017-07-03")])
def test_backtest_sees_only_history_before_each_origin(train_window, start):
    s = _series(kinds={"2025-12-25": "closed"})
    model = Recorder()
    got = backtest(model, s, train_window)
    for (first, last), origin in zip(model.seen, origins(s)):
        assert first == pd.Timestamp(start)
        assert last == origin - pd.Timedelta(minutes=30)
    assert len(got) == 52 * WEEK
    assert (got.groupby("origin").size() == WEEK).all()
    assert (got["ts"] - got["origin"] >= pd.Timedelta(days=7)).all()
    assert (got["forecast"] == 0).all()  # negatives clipped
    # Actuals come from the uncleansed series: the closed day stays zero.
    assert (got.loc[got["ts"].dt.normalize() == "2025-12-25", "actual"] == 0).all()


def test_seasonal_naive_forecast_is_the_week_before_the_origin():
    from src.baselines import SeasonalNaive

    s = _series()
    got = backtest(SeasonalNaive(), s, "primary")
    first = got[got["origin"] == origins(s)[0]]
    last_week = s.loc[origins(s)[0] - pd.Timedelta(days=7): origins(s)[0] - pd.Timedelta(minutes=30), "demand"]
    assert np.array_equal(first["forecast"].to_numpy(), last_week.to_numpy())


def test_score_by_hand():
    ts = pd.date_range("2025-09-01 08:00", periods=4, freq="30min")
    frame = pd.DataFrame({
        "ts": ts,
        "actual": [10.0, 20.0, 30.0, 99.0],
        "forecast": [12.0, 18.0, 36.0, 0.0],
        "is_open": [True, True, True, False],
        "day_kind": "normal",
    })
    got = score(frame, scale=2.0, peak={(False, 16)}).loc["all"]
    assert got["wape"] == pytest.approx(10 / 60)
    assert got["mae"] == pytest.approx(10 / 3)
    assert got["rmse"] == pytest.approx(np.sqrt((4 + 4 + 36) / 3))
    assert got["mase"] == pytest.approx(10 / 3 / 2.0)
    assert got["bias"] == pytest.approx(6 / 60)
    assert got["n"] == 3
    assert score(frame, 2.0, {(False, 16)}).loc["peak", "n"] == 1


def test_peak_keys_ignore_the_test_window():
    s = _series()
    before = peak_keys(s)
    s.loc["2026-01-05", "demand"] = 1e6
    assert peak_keys(s) == before


def test_post_closure_skips_weekend_and_other_closures():
    dates = pd.date_range("2025-12-24", "2026-01-06")
    kind = pd.Series("normal", index=dates)
    kind[["2025-12-25", "2025-12-26", "2026-01-01"]] = "closed"
    days = pd.DataFrame({"kind": kind, "open": kind != "closed"})
    got = post_closure_days(days)
    # Fri 26 closed, so Mon 29 follows the Christmas closures; Fri 2 follows Thu 1.
    assert list(got) == [pd.Timestamp("2025-12-29"), pd.Timestamp("2026-01-02")]
