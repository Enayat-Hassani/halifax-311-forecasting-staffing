"""Forecast half-hourly demand with Seasonal Naive, MSTL, Prophet, XGBoost or Chronos-2.

Model settings are fixed before the test. `context` limits how much history a model reads.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Protocol

import numpy as np
import pandas as pd

WEEK = 336
LAG_WEEKS = (2, 3, 4, 5)
XGB_PARAMS = {
    "objective": "count:poisson",
    "eta": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 0,
}
CHRONOS_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"


class Forecaster(Protocol):
    name: str

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray: ...


class SeasonalNaive:
    """The last observed week, repeated."""

    name = "seasonal_naive"
    context = WEEK

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        return np.resize(history.to_numpy(dtype=float)[-WEEK:], horizon)


class MSTLModel:
    """Model daily and weekly patterns. Primary training has too little data for yearly seasonality."""

    name = "mstl"

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        from statsforecast.models import MSTL

        fitted = MSTL(season_length=[48, WEEK]).fit(history.to_numpy(dtype=float))
        return fitted.predict(h=horizon)["mean"]


class ProphetModel:
    """Multiplicative seasonality, with separate daily shapes for weekdays and weekends."""

    name = "prophet"

    def __init__(self, yearly: bool):
        self.yearly = yearly

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        from prophet import Prophet

        model = Prophet(
            seasonality_mode="multiplicative",
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=self.yearly,
        )
        for name in ("weekday", "weekend"):
            model.add_seasonality(
                f"daily_{name}", period=1, fourier_order=12, condition_name=name
            )
        model.fit(_frame(history.index, history.to_numpy()))
        future = pd.date_range(
            history.index[-1] + pd.Timedelta(minutes=30), periods=horizon, freq="30min"
        )
        return model.predict(_frame(future))["yhat"].to_numpy()


class XGBoostModel:
    """Fit one model per slot with calendar features and lags of at least 14 days."""

    name = "xgboost"

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        import xgboost as xgb

        if horizon > LAG_WEEKS[0] * WEEK:
            raise ValueError(f"horizon {horizon} is past the shortest lag of {LAG_WEEKS[0]} weeks")
        train = lag_features(history, history.index)
        keep = train[[f"lag_{k}w" for k in LAG_WEEKS]].notna().all(axis=1).to_numpy()
        model = xgb.train(
            XGB_PARAMS, xgb.DMatrix(train[keep], history[keep]), num_boost_round=500
        )
        future = pd.date_range(
            history.index[-1] + pd.Timedelta(minutes=30), periods=horizon, freq="30min"
        )
        return model.predict(xgb.DMatrix(lag_features(history, future)))


class Chronos2Model:
    """Run zero-shot Chronos-2. Its point forecast is the median.

    `calendar=True` adds calendar features as known covariates. A worker process keeps
    PyTorch and XGBoost OpenMP runtimes separate on macOS.
    """

    context = 8192
    _worker = None

    def __init__(self, calendar: bool = False):
        self.calendar = calendar
        self.name = "chronos2_calendar" if calendar else "chronos2"

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        if self._worker is None:
            self._worker = ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"))
        history = history.iloc[-self.context :]
        inputs = history.to_numpy(dtype=float)
        if self.calendar:
            future = pd.date_range(
                history.index[-1] + pd.Timedelta(minutes=30), periods=horizon, freq="30min"
            )
            inputs = {
                "target": inputs,
                "past_covariates": _calendar(history.index),
                "future_covariates": _calendar(future),
            }
        return self._worker.submit(_chronos_median, inputs, horizon).result()


_pipeline = None


def _chronos_median(inputs: np.ndarray | dict, horizon: int) -> np.ndarray:
    global _pipeline
    import torch
    from chronos import BaseChronosPipeline

    if _pipeline is None:
        _pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2",
            revision=CHRONOS_REVISION,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    quantiles, _ = _pipeline.predict_quantiles(
        [inputs], prediction_length=horizon, quantile_levels=[0.5]
    )
    return quantiles[0].numpy().ravel()


def lag_features(y: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return calendar and lag features for `index`; missing history stays missing.

    Lags start at 14 days, so they are available at the forecast origin.
    """
    features = pd.DataFrame(_calendar(index), index=index)
    for k in LAG_WEEKS:
        features[f"lag_{k}w"] = y.reindex(index - pd.Timedelta(weeks=k)).to_numpy()
    features["lag_mean"] = features[[f"lag_{k}w" for k in LAG_WEEKS]].mean(axis=1)
    features["lag_52w"] = y.reindex(index - pd.Timedelta(weeks=52)).to_numpy()
    return features


def _calendar(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "slot": (index.hour * 2 + index.minute // 30).to_numpy(),
        "dow": index.dayofweek.to_numpy(),
        "month": index.month.to_numpy(),
    }


def _frame(index: pd.DatetimeIndex, y: np.ndarray | None = None) -> pd.DataFrame:
    weekend = index.dayofweek >= 5
    frame = pd.DataFrame({"ds": index, "weekday": ~weekend, "weekend": weekend})
    if y is not None:
        frame["y"] = y
    return frame
