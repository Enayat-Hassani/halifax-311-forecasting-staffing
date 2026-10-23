"""Forecast weekly agent demand 13 and 26 weeks ahead, then estimate required FTE.

The backtest uses open intervals and replaces outage days. Cost and uplift follow
`src/decide.py`, with weekly demand in place of half-hourly demand.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import XGB_PARAMS, _chronos_median
from src.clean import WINDOWS
from src.decide import RATIOS, TABLE_RATIOS
from src.evaluate import RUNS, cleanse

LEADS = (13, 26)
WEEK = pd.Timedelta(weeks=1)
FTE = {"handle_seconds": 237.0, "occupancy": 0.85, "shrinkage": 0.325, "paid_hours": 37.5}
CALIBRATION = pd.date_range("2024-09-02", "2025-08-25", freq="W-MON")
TEST = pd.date_range(WINDOWS["test"][0], WINDOWS["test"][1] - pd.Timedelta(days=6), freq="W-MON")


def weekly(series: pd.DataFrame) -> pd.DataFrame:
    """Return weekly open-interval demand and the number of closed days."""
    demand = series["demand"].where(series["day_kind"] != "outage", cleanse(series))
    days = series.index.normalize()
    monday = days - pd.to_timedelta(days.dayofweek, unit="D")
    frame = pd.DataFrame({
        "demand": (demand * series["is_open"]).groupby(monday).sum(),
        "closed": (series["day_kind"] == "closed").groupby(monday).sum() // 48,
    })
    frame = frame[frame.index >= WINDOWS["secondary"][0]]
    return frame[(series.groupby(monday).size() == 336).reindex(frame.index)]


def fte(volume, handle_seconds, occupancy, shrinkage, paid_hours):
    return volume * handle_seconds / 3600 / occupancy / (1 - shrinkage) / paid_hours


def _calendar(index: pd.DatetimeIndex, closed) -> dict[str, np.ndarray]:
    return {
        "week": index.isocalendar().week.to_numpy(dtype=float),
        "closed": np.asarray(closed, dtype=float),
    }


class SeasonalNaive:
    name = "seasonal_naive"

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        y = history["demand"]
        return y.reindex(future.index - pd.Timedelta(weeks=52)).to_numpy()


class Prophet:
    name = "prophet"

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        from prophet import Prophet as _Prophet

        model = _Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.add_regressor("closed")
        model.fit(pd.DataFrame({"ds": history.index, "y": history["demand"], "closed": history["closed"]}))
        return model.predict(pd.DataFrame({"ds": future.index, "closed": future["closed"]}))["yhat"].to_numpy()


class XGBoost:
    """Use calendar features and weekly demand lags for each target week."""

    name = "xgboost"

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        lead = len(future)
        y = history["demand"]
        train = self.features(y, history, lead)
        keep = train.notna().all(axis=1).to_numpy()
        model = xgb.train(XGB_PARAMS, xgb.DMatrix(train[keep], y[keep]), num_boost_round=500)
        return model.predict(xgb.DMatrix(self.features(y, future, lead)))

    @staticmethod
    def features(y: pd.Series, frame: pd.DataFrame, lead: int, lags=(0, 1, 2, 3)) -> pd.DataFrame:
        if lead < 1:
            raise ValueError(f"lead {lead} must be at least one week")
        index = frame.index
        out = pd.DataFrame(_calendar(index, frame["closed"]), index=index)
        for k in lags:
            out[f"lag_{lead + k}w"] = y.reindex(index - pd.Timedelta(weeks=lead + k)).to_numpy()
        out["lag_52w"] = y.reindex(index - pd.Timedelta(weeks=52)).to_numpy()
        out["lag_104w"] = y.reindex(index - pd.Timedelta(weeks=104)).to_numpy()
        return out


class Chronos2:
    _worker = None

    def __init__(self, calendar: bool = False):
        self.calendar = calendar
        self.name = "chronos2_calendar" if calendar else "chronos2"

    def forecast(self, history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
        if Chronos2._worker is None:
            Chronos2._worker = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
        inputs = history["demand"].to_numpy(dtype=float)
        if self.calendar:
            inputs = {
                "target": inputs,
                "past_covariates": _calendar(history.index, history["closed"]),
                "future_covariates": _calendar(future.index, future["closed"]),
            }
        return Chronos2._worker.submit(_chronos_median, inputs, len(future)).result()


MODELS = [SeasonalNaive(), Prophet(), XGBoost(), Chronos2(), Chronos2(calendar=True)]


def backtest(model, weeks: pd.DataFrame, lead: int, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per target week: the forecast made `lead` weeks before it."""
    rows = []
    for target in targets:
        origin = target - lead * WEEK
        history = weeks[weeks.index < origin]
        future = weeks.loc[origin:target, ["closed"]]
        rows.append((target, weeks.at[target, "demand"], float(model.forecast(history, future)[-1])))
    return pd.DataFrame(rows, columns=["week", "actual", "forecast"])


def week_ahead(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Sum one-week-ahead forecasts over open intervals."""
    f = forecasts[forecasts["is_open"]]
    week = pd.DatetimeIndex(f["origin"]) + WEEK
    return f.groupby(week)[["actual", "forecast"]].sum().rename_axis("week").reset_index()


def costs(test: pd.DataFrame, calib: pd.DataFrame, params: dict = FTE) -> pd.DataFrame:
    """Return mean weekly FTE cost for point and uplift staffing."""
    need = fte(test["actual"].to_numpy(), **params)
    point = test["forecast"].to_numpy()
    uplift = np.quantile(calib["actual"] / calib["forecast"], RATIOS / (1 + RATIOS))
    out = {}
    for method, staffed in (
        ("point", [fte(point, **params)] * len(RATIOS)),
        ("uplift", [fte(point * u, **params) for u in uplift]),
    ):
        gap = np.array(staffed) - need
        out[method] = (RATIOS[:, None] * np.clip(-gap, 0, None) + np.clip(gap, 0, None)).mean(axis=1)
    return pd.DataFrame(out, index=pd.Index(RATIOS, name="r"))


def main(
    series_path: Path = Path("data/processed/series.parquet"),
    results: Path = Path("results"),
) -> Path:
    import logging

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    weeks = weekly(pd.read_parquet(series_path))
    store = results / "capacity"
    store.mkdir(parents=True, exist_ok=True)

    runs = {}
    for model in MODELS:
        for lead in LEADS:
            frames = []
            for part, targets in (("test", TEST), ("calibration", CALIBRATION)):
                path = store / f"{model.name}_{lead}w_{part}.parquet"
                if not path.exists():
                    backtest(model, weeks, lead, targets).to_parquet(path)
                frames.append(pd.read_parquet(path))
            runs[(model.name, lead)] = frames

    def accuracy(f):
        error = f["forecast"] - f["actual"]
        return error.abs().sum() / f["actual"].sum(), error.sum() / f["actual"].sum()

    one_week = {}
    for model, window in RUNS:
        name = f"{model.name}_{window}"
        path = results / "forecasts" / f"{name}.parquet"
        if path.exists():
            one_week[name] = accuracy(week_ahead(pd.read_parquet(path)))

    lines = [
        "# Capacity horizon",
        "",
        "This report extends the staffing comparison to weekly forecasts 13 and 26 weeks ahead.",
        "52 test weeks, 2025-09-01 to 2026-08-30. Forecasts use the secondary training window "
        "and open intervals. Each forecast starts on Monday, `lead` weeks before the target. "
        "Bias = (forecast − actual) / actual. Prophet, XGBoost and calendar Chronos-2 use "
        "the known count of closed days in each target week; other runs do not.",
        "",
        "## Weekly accuracy at the scheduling lead",
        "",
        "One-week-ahead forecasts, summed over open intervals.",
        "",
        "| Run | WAPE | Bias |",
        "|---|---:|---:|",
    ]
    lines += [f"| {n} | {w:.1%} | {b:+.1%} |" for n, (w, b) in one_week.items()]
    lines += ["", "## Weekly accuracy at the capacity leads", "",
              "| Model | " + " | ".join(f"WAPE {l} w | Bias {l} w" for l in LEADS) + " |",
              "|---|" + "---:|" * 2 * len(LEADS)]
    for model in MODELS:
        cells = [f"{a:.1%} | {b:+.1%}" for a, b in (accuracy(runs[(model.name, l)][0]) for l in LEADS)]
        lines.append(f"| {model.name} | " + " | ".join(cells) + " |")

    demand = weeks.loc[TEST, "demand"]
    lines += [
        "",
        "## FTE",
        "",
        f"FTE = volume × handle time ÷ 3,600 ÷ occupancy ÷ (1 − shrinkage) ÷ paid hours. "
        f"Assumptions: "
        f"{FTE['handle_seconds']:.0f} s handle time, {FTE['occupancy']:.0%} occupancy, "
        f"{FTE['shrinkage']:.1%} shrinkage, and {FTE['paid_hours']} paid hours per week. Actual demand "
        f"equals {fte(demand, **FTE).mean():.1f} FTE a week on average "
        f"({fte(demand, **FTE).min():.1f}–{fte(demand, **FTE).max():.1f}). At the upper bound on "
        f"handle time, 353 s, it equals {fte(demand, **dict(FTE, handle_seconds=353.0)).mean():.1f}. "
        "This estimate excludes interval-level safety staffing and shift inefficiency. "
        "It is not a final headcount plan. The assumptions scale all estimates equally "
        "and do not change rankings.",
        "",
    ]
    for lead in LEADS:
        table = {m.name: costs(*runs[(m.name, lead)]) for m in MODELS}
        columns = [int(np.argmin(np.abs(RATIOS - r))) for r in TABLE_RATIOS]
        lines += [
            f"### Weekly cost at {lead} weeks, FTE",
            "",
            "Mean weekly cost = r × short FTE + spare FTE. Uplift uses the pooled matching "
            "quantile of actual / forecast from 52 calibration weeks at the same lead.",
            "",
            "| Model | Method | " + " | ".join(f"r = {r:g}" for r in TABLE_RATIOS) + " |",
            "|---|---|" + "---:|" * len(TABLE_RATIOS),
        ]
        for name, c in table.items():
            for method in ("point", "uplift"):
                lines.append(f"| {name} | {method} | " + " | ".join(
                    f"{c[method].iloc[j]:.2f}" for j in columns) + " |")
        lines.append("")
    out = results / "capacity.md"
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    main()
