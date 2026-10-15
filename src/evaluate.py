"""Weekly rolling-origin backtest and the metrics every model is scored on.

Each Monday 00:00 origin forecasts the Monday-Sunday week that starts seven days later:
the forecast a schedule is built from. Models see closure-cleansed history from their
training-window start up to the origin. Only open intervals of the target week are scored.
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
    # In date order, so a day copied from is already cleansed and closures chain back.
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


RUNS = [
    (SeasonalNaive(), "primary"),
    (MSTLModel(), "primary"),
    (MSTLModel(), "secondary"),
    (ProphetModel(yearly=False), "primary"),
    (ProphetModel(yearly=True), "secondary"),
    (XGBoostModel(), "primary"),
    (XGBoostModel(), "secondary"),
    (Chronos2Model(), "primary"),
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

    tables, costs = [], []
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
        seconds = forecasts.groupby("origin")["seconds"].first()
        costs.append((model.name, label, seconds.iloc[0], seconds.median(), seconds.sum()))
    results = pd.concat(tables).rename_axis("split").reset_index()

    lines = [
        "# Baselines",
        "",
        "Generated by `src/evaluate.py`. 52 weekly origins; each forecasts the Monday-Sunday "
        "week that starts seven days later, 2025-09-01 → 2026-08-30. Open intervals only. "
        "Bias is (forecast − actual) / actual, so a positive bias over-forecasts. "
        f"MASE is scaled by the in-sample Seasonal Naive MAE on primary training ({scale:.2f} calls). "
        "History weeks are the weeks a model reads at the last origin.",
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
