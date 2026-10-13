"""Regular half-hourly agent-demand series on local wall-clock time.

From 2024-07-17 the source writes each interval as a fixed UTC+4 timestamp. The repair
reads those labels as UTC+4 and converts them to America/Halifax, which moves the date
and the hour together. From 2023-05-23 to 2024-07-16 the source counts most days' agent
calls twice; those days are halved. Absent intervals are zero. Each slot carries whether
agents were scheduled and the kind of day from `data/special_days.csv`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TIME_BASE_BREAK = pd.Timestamp("2024-07-17")
DUPLICATED = (pd.Timestamp("2023-05-23"), pd.Timestamp("2024-07-16"))
HOLIDAY_CLOSURES_FROM = pd.Timestamp("2023-05-22")
WINDOWS = {
    "secondary": (pd.Timestamp("2017-07-03"), pd.Timestamp("2025-08-31")),
    "primary": (TIME_BASE_BREAK, pd.Timestamp("2025-08-31")),
    "test": (pd.Timestamp("2025-09-01"), pd.Timestamp("2026-08-31")),
}
SOURCE_COLUMNS = {
    "HANDLED": "handled",
    "ABANDONED": "abandoned",
    "OFFERED": "offered",
    "PROCESSED_IN_IVR": "ivr",
    "TOTAL_TALK_TIME": "talk_seconds",
}


def local_time(raw: pd.DataFrame) -> pd.Series:
    """Local wall-clock start of each raw row's interval, naive."""
    # CALL_DATE is midnight US Pacific, so the UTC date of the stamp is the label date.
    day = pd.to_datetime(raw["CALL_DATE"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()
    minute = raw["INTERVAL"].str.extract(r"^\d{2}:(\d{2})")[0].astype(int)
    label = day + pd.to_timedelta(raw["MILITARY_HOUR"] * 60 + minute, unit="min")
    repaired = (
        (label - pd.Timedelta(hours=4))
        .dt.tz_localize("UTC")
        .dt.tz_convert("America/Halifax")
        .dt.tz_localize(None)
    )
    return label.where(label < TIME_BASE_BREAK, repaired).rename("ts")


def undouble(counts: pd.DataFrame) -> pd.DataFrame:
    """Halve handled, abandoned and talk time on days the source counted twice.

    A doubled day has only even counts, apart from a stray interval. Inside DUPLICATED, a
    day is doubled when under a fifth of its non-zero counts are odd; a day counted once has
    about half odd. `offered` keeps the IVR calls, which are not doubled.
    """
    dates = counts.index.normalize()
    calls = counts[["handled", "abandoned"]]
    odd = (calls % 2).sum(axis=1).groupby(dates).sum()
    share = odd / (calls > 0).sum(axis=1).groupby(dates).sum()
    inside = (dates >= DUPLICATED[0]) & (dates <= DUPLICATED[1])
    doubled = inside & (share.reindex(dates).to_numpy() < 0.2)

    out = counts.astype(float)
    half = out.loc[doubled, ["handled", "abandoned", "talk_seconds"]] / 2
    out.loc[doubled, half.columns] = half
    out.loc[doubled, "offered"] -= half["handled"] + half["abandoned"]
    return out


def _is_open(index: pd.DatetimeIndex, closed: pd.DatetimeIndex) -> np.ndarray:
    """Published 311 hours: weekdays 08:00-20:00, weekends 09:00-17:30, closed days off.

    Valid from 2023-05-22. Earlier hours and holiday practice differ.
    """
    minutes = index.hour * 60 + index.minute
    weekday = index.dayofweek < 5
    hours = np.where(
        weekday, (minutes >= 480) & (minutes < 1200), (minutes >= 540) & (minutes < 1050)
    )
    return hours & ~index.normalize().isin(closed)


def build_series(raw: pd.DataFrame, special_days: pd.DataFrame) -> pd.DataFrame:
    """One row per 30-minute local slot, from the first to the last pulled day."""
    ts = local_time(raw)
    counts = undouble(raw[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS).groupby(ts).sum())
    index = pd.date_range(
        ts.min().normalize(),
        ts.max().normalize() + pd.Timedelta(minutes=23 * 60 + 30),
        freq="30min",
        name="ts",
    )
    series = counts.reindex(index, fill_value=0)
    series.insert(0, "demand", series["handled"] + series["abandoned"])

    dates = index.normalize()
    kinds = special_days.set_index("date")["kind"]
    series["is_open"] = _is_open(index, pd.DatetimeIndex(kinds.index[kinds == "closed"]))
    series["day_kind"] = kinds.reindex(dates).fillna("normal").to_numpy()
    return series


def window(series: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows of `series` whose local date is inside the named window, both ends inclusive."""
    start, end = WINDOWS[name]
    dates = series.index.normalize()
    return series[(dates >= start) & (dates <= end)]


def main(
    raw_dir: Path = Path("data/raw"), out: Path = Path("data/processed/series.parquet")
) -> Path:
    raw = pd.read_parquet(sorted(raw_dir.glob("call_volumes_*.parquet"))[-1])
    special_days = pd.read_csv(Path("data/special_days.csv"), parse_dates=["date"])
    out.parent.mkdir(parents=True, exist_ok=True)
    build_series(raw, special_days).to_parquet(out)
    return out


if __name__ == "__main__":
    main()
