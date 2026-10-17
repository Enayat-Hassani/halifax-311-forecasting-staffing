"""Figures for the README, drawn from the snapshot, the series and the backtest files.

Nothing is fitted here. Staffing reuses the primary scenario and the calibration of
`src/decide.py`.
"""

from pathlib import Path

import matplotlib
import matplotlib.dates

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clean import DUPLICATED, local_time
from src.decide import PRIMARY, RATIOS, WEEKS, _lookup, _ratios, poisson_floor
from src.evaluate import RUNS, peak_keys

NAMES = {
    "seasonal_naive_primary": "Seasonal Naive",
    "mstl_primary": "MSTL (primary)",
    "mstl_secondary": "MSTL (secondary)",
    "prophet_primary": "Prophet (primary)",
    "prophet_secondary": "Prophet (secondary)",
    "xgboost_primary": "XGBoost (primary)",
    "xgboost_secondary": "XGBoost (secondary)",
    "chronos2_primary": "Chronos-2",
}
SHOWN = ("xgboost_secondary", "mstl_primary", "chronos2_primary")
COLORS = {"xgboost_secondary": "#1f77b4", "mstl_primary": "#7f7f7f", "chronos2_primary": "#d62728"}
# Label offsets in points: four runs sit within 1.5 agent-hours of each other, and
# Seasonal Naive is at the right edge.
OFFSETS = {
    "mstl_primary": (6, -10),
    "mstl_secondary": (6, 3),
    "prophet_secondary": (-6, 3),
    "chronos2_primary": (6, -10),
    "seasonal_naive_primary": (-6, -3),
}
RIGHT_ALIGNED = ("prophet_secondary", "seasonal_naive_primary")


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def repair(raw: pd.DataFrame, series: pd.DataFrame, path: Path) -> None:
    """Daily agent-queue volume as published and repaired, 28-day mean."""
    published = (raw["HANDLED"] + raw["ABANDONED"]).groupby(local_time(raw).dt.normalize()).sum()
    repaired = series["demand"].resample("D").sum()
    span = slice("2022-01-01", "2025-08-31")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(*DUPLICATED, color="#d62728", alpha=0.08, label="Counted twice")
    ax.plot(published[span].rolling(28).mean(), color="#d62728", label="As published")
    ax.plot(repaired[span].rolling(28).mean(), color="#1f77b4", label="Repaired")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.set_ylabel("Agent-queue calls per day, 28-day mean")
    ax.set_ylim(0)
    ax.set_title("Published 311 agent-queue volume doubled for 14 months")
    ax.legend(loc="lower left")
    _save(fig, path)


def week(runs: dict[str, pd.DataFrame], path: Path) -> None:
    """The test week of median XGBoost WAPE: actual against XGBoost and Chronos-2."""
    xgb = runs["xgboost_secondary"]
    open_ = xgb[xgb["is_open"]]
    error = (open_["forecast"] - open_["actual"]).abs().groupby(open_["origin"]).sum()
    by_week = error / open_.groupby("origin")["actual"].sum()
    origin = by_week.sort_values().index[len(by_week) // 2]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    shown = xgb[xgb["origin"] == origin]
    ax.plot(shown["ts"], shown["actual"], color="black", lw=1, label="Actual")
    for file in ("xgboost_secondary", "chronos2_primary"):
        f = runs[file][runs[file]["origin"] == origin]
        ax.plot(f["ts"], f["forecast"], color=COLORS[file], lw=1.2, label=NAMES[file])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d %b"))
    ax.set_ylabel("Agent calls per half-hour")
    ax.set_title(f"A typical test week, forecast on {origin:%d %b %Y}, 7 days before it starts")
    ax.legend()
    _save(fig, path)


def accuracy_staffing(runs: dict[str, pd.DataFrame], test: pd.DataFrame, path: Path) -> None:
    """WAPE against weekly net staffing error, with the simulated Poisson floor."""
    lookup = _lookup(PRIMARY)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for file, f in runs.items():
        f = f[f["is_open"]]
        actual, point = f["actual"].to_numpy(), f["forecast"].to_numpy()
        wape = np.abs(point - actual).sum() / actual.sum()
        error = 0.5 * np.abs(lookup(point) - lookup(actual)).sum() / WEEKS
        ax.scatter(wape, error, color=COLORS.get(file, "#444444"))
        ax.annotate(NAMES[file], (wape, error), fontsize=8, xytext=OFFSETS.get(file, (6, -3)),
                    textcoords="offset points", ha="right" if file in RIGHT_ALIGNED else "left")
    floor = poisson_floor(test)
    ax.scatter(floor["wape"], floor["short"] + floor["over"], marker="x", color="black")
    ax.annotate("Poisson floor: observed volume used as the rate",
                (floor["wape"], floor["short"] + floor["over"]),
                fontsize=8, xytext=(6, -3), textcoords="offset points")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("WAPE (lower is better)")
    ax.set_ylabel("Net staffing error, agent-hours per week")
    ax.set_title("WAPE and net staffing error produce similar rankings")
    _save(fig, path)


def bias_by_month(runs: dict[str, pd.DataFrame], path: Path) -> None:
    """Monthly bias on open slots: (forecast - actual) / actual."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="black", lw=0.8)
    for file in SHOWN:
        f = runs[file][runs[file]["is_open"]]
        month = f.groupby(f["ts"].dt.to_period("M"))[["forecast", "actual"]].sum()
        bias = month["forecast"] / month["actual"] - 1
        ax.plot(bias.index.to_timestamp(), bias, marker="o", color=COLORS[file], label=NAMES[file])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.set_ylabel("Monthly bias (above 0 = over-forecast)")
    ax.set_title("Monthly bias differs by model and training window")
    ax.legend()
    _save(fig, path)


def buffer(runs: dict[str, pd.DataFrame], calib: dict[str, pd.DataFrame], peak: set, path: Path) -> None:
    """Weekly cost by cost ratio: range over the eight runs, point against uplift staffing."""
    lookup = _lookup(PRIMARY)
    levels = RATIOS / (1 + RATIOS)
    cost = {"point": [], "uplift": []}
    for file, f in runs.items():
        f = f[f["is_open"]]
        actual, point = f["actual"].to_numpy(), f["forecast"].to_numpy()
        need, staffed = lookup(actual), lookup(point)
        uplift = np.quantile(_ratios(calib[file], peak).to_numpy(), levels)
        for method, rows in cost.items():
            gaps = [staffed - need if method == "point" else lookup(point * u) - need for u in uplift]
            rows.append([
                0.5 * (r * np.clip(-g, 0, None) + np.clip(g, 0, None)).sum() / WEEKS
                for r, g in zip(RATIOS, gaps)
            ])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, color, label in (
        ("point", "#d62728", "Staffed to the point forecast"),
        ("uplift", "#1f77b4", "Point forecast with calibrated uplift"),
    ):
        c = np.array(cost[method])
        ax.fill_between(RATIOS, c.min(axis=0), c.max(axis=0), color=color, alpha=0.25)
        ax.plot(RATIOS, c.min(axis=0), color=color, label=label)
    ax.axvline(1, color="black", lw=0.8, ls=":")
    ax.annotate("equal costs", (1, 0.5), xycoords=("data", "axes fraction"), fontsize=8,
                xytext=(4, 0), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.set_xlabel("Cost ratio r: one short agent-hour costs r spare agent-hours")
    ax.set_ylabel("Modelled weekly penalty, spare agent-hours")
    ax.set_title("Forecast uplift reduces penalty as understaffing cost rises")
    ax.legend(loc="upper left", title="Line: lowest penalty. Band: range across 8 runs",
              title_fontsize=8, fontsize=9)
    _save(fig, path)


def main(
    raw_path: Path = Path("data/raw/call_volumes_2026-09-13.parquet"),
    series_path: Path = Path("data/processed/series.parquet"),
    results: Path = Path("results"),
) -> Path:
    out = results / "figures"
    out.mkdir(exist_ok=True)
    series = pd.read_parquet(series_path)
    files = [f"{model.name}_{window}" for model, window in RUNS]
    runs = {f: pd.read_parquet(results / "forecasts" / f"{f}.parquet") for f in files}
    calib = {f: pd.read_parquet(results / "calibration" / f"{f}.parquet") for f in files}

    repair(pd.read_parquet(raw_path), series, out / "repair.png")
    week(runs, out / "week.png")
    accuracy_staffing(runs, runs["xgboost_secondary"], out / "accuracy_staffing.png")
    bias_by_month(runs, out / "bias_by_month.png")
    buffer(runs, calib, peak_keys(series), out / "buffer.png")
    return out


if __name__ == "__main__":
    main()
