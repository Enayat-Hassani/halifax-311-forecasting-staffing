"""Cleaning tests. The snapshot tests need a pulled snapshot."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean import (
    DUPLICATED,
    HOLIDAY_CLOSURES_FROM,
    WINDOWS,
    build_series,
    local_time,
    undouble,
    window,
)

NO_SPECIAL_DAYS = pd.DataFrame({"date": pd.to_datetime([]), "kind": [], "note": []})


def _raw(rows: list[tuple[str, int, int, int]]) -> pd.DataFrame:
    """Raw-shaped rows from (label date, hour, minute, handled), as the source writes them."""
    out = []
    for date, hour, minute, handled in rows:
        start = pd.Timestamp(date) + pd.Timedelta(hours=hour, minutes=minute)
        end = start + pd.Timedelta(minutes=30)
        out.append(
            {
                # Midnight US Pacific, as the service stamps it.
                "CALL_DATE": int(pd.Timestamp(f"{date} 07:00", tz="UTC").timestamp() * 1000),
                "MILITARY_HOUR": hour,
                "INTERVAL": f"{start:%I:%M %p} - {end:%I:%M %p}",
                "OFFERED": handled,
                "HANDLED": handled,
                "ABANDONED": 1,
                "PROCESSED_IN_IVR": 0,
                "TOTAL_TALK_TIME": 240 * handled,
            }
        )
    return pd.DataFrame(out)


def test_pre_break_label_is_local_time():
    got = local_time(_raw([("2024-07-16", 10, 30, 5)]))
    assert got.iloc[0] == pd.Timestamp("2024-07-16 10:30")


def test_post_break_label_is_read_as_utc_plus_4():
    got = local_time(_raw([("2024-08-07", 14, 0, 5), ("2025-01-15", 14, 0, 5)]))
    assert list(got) == [pd.Timestamp("2024-08-07 07:00"), pd.Timestamp("2025-01-15 06:00")]


def test_post_break_date_moves_with_the_hour():
    # 2024-08-10 is a Saturday; its 01:00 label is Friday evening in Halifax.
    got = local_time(_raw([("2024-08-10", 1, 0, 5)]))
    assert got.iloc[0] == pd.Timestamp("2024-08-09 18:00")


def test_autumn_changeover_rows_are_summed():
    # 2024-11-03 08:00 and 09:00 at UTC+4 are 01:00 ADT and 01:00 AST in Halifax.
    series = build_series(
        _raw([("2024-11-03", 8, 0, 3), ("2024-11-03", 9, 0, 4)]), NO_SPECIAL_DAYS
    )
    assert series.loc["2024-11-03 01:00", "handled"] == 7


def _counts(day: str, handled: list[int]) -> pd.DataFrame:
    index = pd.date_range(f"{day} 08:00", periods=len(handled), freq="30min")
    h = np.array(handled)
    return pd.DataFrame(
        {"handled": h, "abandoned": 2, "offered": 2 * h + 10, "ivr": 6, "talk_seconds": 480 * h},
        index=index,
    )


def test_undouble_halves_even_days_inside_the_period_only():
    even = [20, 34, 18, 40, 26]
    got = undouble(pd.concat([_counts("2023-10-04", even), _counts("2023-04-05", even)]))
    inside, outside = got.loc["2023-10-04"], got.loc["2023-04-05"]
    assert list(inside["handled"]) == [10, 17, 9, 20, 13]
    assert (inside["abandoned"] == 1).all()
    assert list(inside["talk_seconds"]) == [480 * h for h in [10, 17, 9, 20, 13]]
    # Offered drops by the calls removed; the IVR calls stay.
    assert list(inside["offered"]) == [2 * h + 10 - (h // 2 + 1) for h in even]
    assert list(outside["handled"]) == even


def test_undouble_keeps_a_day_counted_once_and_a_stray_odd_interval():
    once = [21, 34, 17, 41, 26]
    stray = [20, 34, 18, 40, 26, 30, 44, 21]
    got = undouble(pd.concat([_counts("2023-10-04", once), _counts("2023-10-05", stray)]))
    assert list(got.loc["2023-10-04", "handled"]) == once
    assert list(got.loc["2023-10-05", "handled"]) == [h / 2 for h in stray]


def test_absent_slots_are_zero_and_index_is_regular():
    series = build_series(
        _raw([("2024-03-04", 9, 0, 5), ("2024-03-05", 9, 0, 5)]), NO_SPECIAL_DAYS
    )
    assert len(series) == 96
    assert series.index.is_unique
    assert (series.index.to_series().diff().dropna() == pd.Timedelta(minutes=30)).all()
    assert series.loc["2024-03-04 09:30", "demand"] == 0
    assert series.loc["2024-03-04 09:00", "demand"] == 6


def test_is_open_follows_published_hours_and_closed_days():
    # 2025-03-03 is a Monday, 2025-03-08 a Saturday, 2025-03-04 listed as closed.
    closed = pd.DataFrame(
        {"date": pd.to_datetime(["2025-03-04"]), "kind": ["closed"], "note": ["x"]}
    )
    series = build_series(
        _raw([("2025-03-03", 12, 0, 1), ("2025-03-08", 12, 0, 1)]), closed
    )
    is_open = series["is_open"]
    assert is_open["2025-03-03 08:00"] and is_open["2025-03-03 19:30"]
    assert not is_open["2025-03-03 07:30"] and not is_open["2025-03-03 20:00"]
    assert is_open["2025-03-08 09:00"] and is_open["2025-03-08 17:00"]
    assert not is_open["2025-03-08 08:30"] and not is_open["2025-03-08 17:30"]
    assert not is_open["2025-03-04"].any()
    assert (series.loc["2025-03-04", "day_kind"] == "closed").all()
    assert int(is_open["2025-03-03"].sum()) == 24 and int(is_open["2025-03-08"].sum()) == 17


def test_windows_are_inclusive_and_test_does_not_overlap_training():
    index = pd.date_range("2017-01-01", "2026-09-12 23:30", freq="30min", name="ts")
    series = pd.DataFrame({"demand": 0}, index=index)
    for name, (start, end) in WINDOWS.items():
        dates = window(series, name).index.normalize()
        assert dates.min() == start and dates.max() == end
    assert WINDOWS["test"][0] > WINDOWS["primary"][1]
    assert WINDOWS["test"][0] > WINDOWS["secondary"][1]


# Snapshot checks.

RAW_DIR = Path("data/raw")


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    matches = sorted(RAW_DIR.glob("call_volumes_*.parquet"))
    if not matches:
        pytest.fail("no frozen snapshot in data/raw; run src.ingest.pull() first")
    return pd.read_parquet(matches[-1])


@pytest.fixture(scope="module")
def series(raw) -> pd.DataFrame:
    return build_series(raw, pd.read_csv("data/special_days.csv", parse_dates=["date"]))


def test_series_keeps_every_agent_call_outside_the_duplicated_period(raw, series):
    dates = local_time(raw).dt.normalize()
    keep = (dates < DUPLICATED[0]) | (dates > DUPLICATED[1])
    outside = (series.index.normalize() < DUPLICATED[0]) | (series.index.normalize() > DUPLICATED[1])
    assert series.loc[outside, "demand"].sum() == (raw["HANDLED"] + raw["ABANDONED"])[keep].sum()


@pytest.mark.parametrize("boundary", ["2023-05-23", "2024-07-17"])
def test_weekday_level_is_continuous_across_the_duplicated_period(series, boundary):
    day = pd.Timestamp(boundary)
    def level(start, end):
        part = series.loc[start:end]
        part = part[(part.index.dayofweek < 5) & (part["day_kind"] == "normal")]
        return part["demand"].groupby(part.index.normalize()).sum().mean()
    before = level(day - pd.Timedelta(days=28), day - pd.Timedelta(minutes=30))
    after = level(day, day + pd.Timedelta(days=28) - pd.Timedelta(minutes=30))
    assert 0.8 < after / before < 1.25


def _weekday_profile(series: pd.DataFrame, start: str, end: str) -> np.ndarray:
    part = series.loc[start:end]
    part = part[part.index.dayofweek < 5]
    return part.groupby(part.index.hour * 2 + part.index.minute // 30)["demand"].mean().to_numpy()


def test_repaired_profile_aligns_with_pre_break_profile(series):
    before = _weekday_profile(series, "2022-07-01", "2023-04-30")
    after = _weekday_profile(series, "2024-08-01", "2025-06-30")
    corr = {s: np.corrcoef(before, np.roll(after, s))[0, 1] for s in (-2, -1, 0, 1, 2)}
    assert corr[0] > 0.99
    assert corr[0] == max(corr.values())


def test_closed_days_match_days_without_agent_calls(series):
    since = series.loc[HOLIDAY_CLOSURES_FROM : WINDOWS["test"][1] + pd.Timedelta(hours=23)]
    minutes = since.index.hour * 60 + since.index.minute
    weekday = since.index.dayofweek < 5
    scheduled = np.where(
        weekday, (minutes >= 480) & (minutes < 1200), (minutes >= 540) & (minutes < 1050)
    )
    per_day = since["demand"].where(scheduled, 0).groupby(since.index.normalize()).sum()
    listed = since.loc[since["day_kind"] == "closed"].index.normalize().unique()
    assert set(per_day[per_day <= 5].index) == set(listed)


def test_open_intervals_in_test_window_are_rarely_absent(raw, series):
    test = window(series, "test")
    open_slots = test.index[test["is_open"]]
    recorded = pd.DatetimeIndex(local_time(raw)).unique()
    absent = (~open_slots.isin(recorded)).mean()
    assert absent < 0.005
