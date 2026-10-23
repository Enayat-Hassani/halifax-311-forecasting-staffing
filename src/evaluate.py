"""Backtest weekly forecasts and score open intervals in each target week.

Each Monday origin forecasts the week that starts seven days later. Models use cleansed
history from their training-window start through the origin.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import (
    WEEK,
    Chronos2Model,
    Forecaster,
    MSTLModel,
    ProphetModel,
    SeasonalNaive,
    XGBoostModel,
)
from src.clean import WINDOWS, window

LEAD = pd.Timedelta(days=7)
TARGET = pd.Timedelta(days=7)
SLOT = pd.Timedelta(minutes=30)
REPLACED = ("closed", "outage")


def cleanse(series: pd.DataFrame) -> pd.Series:
    """Demand with closed and outage days replaced by the last valid same-weekday day."""
    days = series["demand"].to_numpy(dtype=float).reshape(-1, 48).copy()
    kinds = series["day_kind"].to_numpy()[::48]
    # Process dates in order so consecutive closures use the prior repaired day.
    for i in np.flatnonzero(np.isin(kinds, REPLACED)):
        if i >= 7:
            days[i] = days[i - 7]
    return pd.Series(days.ravel(), index=series.index, name="demand")


def origins(series: pd.DataFrame) -> pd.DatetimeIndex:
    """Monday 00:00 origins whose whole target week falls inside the test window."""
    start, end = WINDOWS["test"]
    weeks = pd.date_range(start, end - TARGET + pd.Timedelta(days=1), freq="W-MON")
    return weeks - LEAD


def backtest(
    model: Forecaster,
    series: pd.DataFrame,
    train_window: str,
    at: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """One row per target slot: origin, ts, actual, forecast, is_open, day_kind, seconds.

    Runs the origins `at`, by default the test origins. `seconds` is the wall time of the
    origin's `forecast` call.
    """
    history_all = cleanse(series)
    start = WINDOWS[train_window][0]
    horizon = (LEAD + TARGET) // SLOT
    frames = []
    for origin in origins(series) if at is None else at:
        history = history_all.loc[start : origin - SLOT]
        target = pd.date_range(origin + LEAD, periods=WEEK, freq="30min")
        started = time.perf_counter()
        forecast = model.forecast(history, horizon)
        seconds = time.perf_counter() - started
        forecast = np.clip(forecast[-WEEK:], 0.0, None)
        rows = series.loc[target, ["demand", "is_open", "day_kind"]]
        frames.append(
            rows.rename(columns={"demand": "actual"})
            .assign(origin=origin, forecast=forecast, seconds=seconds)
            .rename_axis("ts")
            .reset_index()
        )
    columns = ["origin", "ts", "actual", "forecast", "is_open", "day_kind", "seconds"]
    return pd.concat(frames, ignore_index=True)[columns]


def slot_key(ts: pd.DatetimeIndex) -> list[tuple[bool, int]]:
    return list(zip(ts.dayofweek >= 5, ts.hour * 2 + ts.minute // 30))


def peak_keys(series: pd.DataFrame) -> set[tuple[bool, int]]:
    """(is_weekend, slot) pairs in the top quartile of mean demand, primary training."""
    primary = window(series, "primary")
    normal = primary[primary["is_open"] & (primary["day_kind"] == "normal")]
    ts = normal.index
    means = normal["demand"].groupby([ts.dayofweek >= 5, ts.hour * 2 + ts.minute // 30]).mean()
    return {(bool(w), int(s)) for w, s in means.index[means >= means.quantile(0.75)]}


def mase_scale(series: pd.DataFrame) -> float:
    """In-sample MAE of Seasonal Naive at m = 336 on cleansed primary training, open slots."""
    primary = window(series, "primary")
    y = window(cleanse(series).to_frame(), "primary")["demand"]
    error = (y - y.shift(WEEK)).abs()
    return float(error[primary["is_open"] & error.notna()].mean())


def post_closure_days(days: pd.DataFrame) -> pd.DatetimeIndex:
    """First open weekday after each closed day. `days` has columns kind, open by date."""
    found, waiting = [], False
    for day, row in days.sort_index().iterrows():
        if row["kind"] == "closed":
            waiting = True
        elif waiting and row["open"] and day.dayofweek < 5:
            found.append(day)
            waiting = False
    return pd.DatetimeIndex(found)


def splits(forecasts: pd.DataFrame, peak: set) -> dict[str, np.ndarray]:
    """Row masks for the splits all, peak, off_peak and post_closure."""
    ts = pd.DatetimeIndex(forecasts["ts"])
    days = (
        forecasts.assign(date=ts.normalize())
        .groupby("date")
        .agg(kind=("day_kind", "first"), open=("is_open", "any"))
    )
    is_peak = np.array([k in peak for k in slot_key(ts)])
    return {
        "all": np.ones(len(forecasts), bool),
        "peak": is_peak,
        "off_peak": ~is_peak,
        "post_closure": ts.normalize().isin(post_closure_days(days)),
    }


def score(forecasts: pd.DataFrame, scale: float, peak: set) -> pd.DataFrame:
    """One row per split: all, peak, off_peak, post_closure."""
    rows = {}
    for name, mask in splits(forecasts, peak).items():
        f = forecasts[mask & forecasts["is_open"].to_numpy()]
        error = f["forecast"] - f["actual"]
        total = f["actual"].sum()
        rows[name] = {
            "wape": error.abs().sum() / total,
            "mae": error.abs().mean(),
            "rmse": float(np.sqrt((error**2).mean())),
            "mase": error.abs().mean() / scale,
            "bias": error.sum() / total,
            "n": len(f),
        }
    return pd.DataFrame(rows).T


def breakdown(forecasts: pd.DataFrame) -> pd.DataFrame:
    """WAPE and bias of open slots by test month and by day type.

    A day is a surge or a lull when its actual demand is in the top or bottom tenth of the
    test year, relative to the median of its weekday. Bands use actuals only, so every run
    is scored on the same days.
    """
    f = forecasts[forecasts["is_open"]]
    ts = pd.DatetimeIndex(f["ts"])
    day = f.groupby(ts.normalize())["actual"].sum()
    ratio = day / day.groupby(day.index.dayofweek).transform("median")
    rank = ratio.rank(pct=True)
    band = pd.Series(np.select([rank > 0.9, rank <= 0.1], ["surge", "lull"], "normal"), ratio.index)
    keys = {"month": ts.strftime("%Y-%m"), "day": band.reindex(ts.normalize()).to_numpy()}
    error = (f["forecast"] - f["actual"]).to_numpy()
    sums = pd.DataFrame({"abs": np.abs(error), "signed": error, "actual": f["actual"].to_numpy()})
    rows = []
    for by, key in keys.items():
        g = sums.groupby(np.asarray(key)).sum()
        rows.append(pd.DataFrame(
            {"by": by, "group": g.index, "wape": g["abs"] / g["actual"], "bias": g["signed"] / g["actual"]}
        ))
    return pd.concat(rows, ignore_index=True)


RUNS = [
    (SeasonalNaive(), "primary"),
    (MSTLModel(), "primary"),
    (MSTLModel(), "secondary"),
    (ProphetModel(yearly=False), "primary"),
    (ProphetModel(yearly=True), "secondary"),
    (XGBoostModel(), "primary"),
    (XGBoostModel(), "secondary"),
    (Chronos2Model(), "primary"),
    (Chronos2Model(calendar=True), "primary"),
]


def main(
    series_path: Path = Path("data/processed/series.parquet"),
    out_dir: Path = Path("results"),
) -> Path:
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet.plot").setLevel(logging.ERROR)
    series = pd.read_parquet(series_path)
    scale, peak = mase_scale(series), peak_keys(series)
    last_origin = origins(series)[-1]

    tables, costs, parts = [], [], []
    for model, train_window in RUNS:
        path = out_dir / "forecasts" / f"{model.name}_{train_window}.parquet"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            backtest(model, series, train_window).to_parquet(path)
        forecasts = pd.read_parquet(path)
        context = getattr(model, "context", None)
        weeks = (last_origin - WINDOWS[train_window][0]).days // 7
        if context:
            weeks = min(weeks, context // WEEK)
        label = "any" if context else train_window
        tables.append(
            score(forecasts, scale, peak).assign(model=model.name, window=label, history_weeks=weeks)
        )
        parts.append(breakdown(forecasts).assign(run=f"{model.name} ({label})"))
        seconds = forecasts.groupby("origin")["seconds"].first()
        costs.append((model.name, label, seconds.iloc[0], seconds.median(), seconds.sum()))
    results = pd.concat(tables).rename_axis("split").reset_index()

    lines = [
        "# Baselines",
        "",
        "52 weekly origins forecast the week starting seven days later, 2025-09-01 to "
        "2026-08-30. Open intervals only. Bias = (forecast − actual) / actual. "
        f"MASE uses the primary-training Seasonal Naive MAE ({scale:.2f} calls). "
        "History weeks show how much history each model reads at the final origin. "
        "The decision analysis compares WAPE rankings with staffing error and cost. "
        "It tests cases where understaffing costs more.",
        "",
    ]
    for split, table in results.groupby("split", sort=False):
        lines += [
            f"## {split.replace('_', ' ').capitalize()} ({int(table['n'].iloc[0]):,} intervals)",
            "",
            "| Model | Window | History weeks | WAPE | MAE | RMSE | MASE | Bias |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in table.itertuples():
            lines.append(
                f"| {r.model} | {r.window} | {r.history_weeks} | {r.wape:.1%} | {r.mae:.2f} | "
                f"{r.rmse:.2f} | {r.mase:.3f} | {r.bias:+.1%} |"
            )
        lines.append("")
    parts = pd.concat(parts)
    runs = list(dict.fromkeys(parts["run"]))
    for by, title, note in (
        ("month", "By month", "Seasonal bias is hidden in the annual total."),
        ("day", "By day type",
         "A surge or lull day is in the top or bottom tenth of the test year relative to "
         "its weekday median. Sorting on actuals pushes surge-day bias down for any forecast."),
    ):
        table = parts[parts["by"] == by]
        lines += [f"## {title}: WAPE (bias)", "", note, "",
                  "| | " + " | ".join(runs) + " |", "|---|" + "---:|" * len(runs)]
        for group, rows in table.groupby("group", sort=True):
            cells = rows.set_index("run").loc[runs]
            lines.append(f"| {group} | " + " | ".join(
                f"{r.wape:.1%} ({r.bias:+.1%})" for r in cells.itertuples()) + " |")
        lines.append("")
    lines += [
        "## Run cost",
        "",
        "Wall time of one forecast call on an Apple M2 CPU, comparable between models only. "
        "The first call of Chronos-2 includes loading the model.",
        "",
        "| Model | Window | First (s) | Median (s) | 52 origins (min) |",
        "|---|---|---:|---:|---:|",
    ]
    for name, label, first, median, total in costs:
        lines.append(f"| {name} | {label} | {first:.1f} | {median:.1f} | {total / 60:.1f} |")
    lines.append("")
    out = out_dir / "baselines.md"
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    main()
