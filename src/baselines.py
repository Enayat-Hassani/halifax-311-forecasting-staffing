"""Seasonal Naive, MSTL, Prophet and XGBoost behind one interface.

Each model fits on the history it is given and forecasts the `horizon` slots that follow
it. Settings are fixed in advance; none is tuned on the test window.
"""

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


class Forecaster(Protocol):
    name: str

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray: ...


class SeasonalNaive:
    """The last observed week, repeated."""

    name = "seasonal_naive"

    def forecast(self, history: pd.Series, horizon: int) -> np.ndarray:
        return np.resize(history.to_numpy(dtype=float)[-WEEK:], horizon)


class MSTLModel:
    """Daily and weekly seasons. One annual cycle in primary training is too few for a third."""

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
    """One direct model for every slot, on calendar and same-slot lags of 14 days or more."""

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


def lag_features(y: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar and lag features for `index`, taken from `y` only.

    Every lag is at least 14 days, so the features of a slot are known at any origin up to
    14 days before it. Lags that fall outside `y` are missing.
    """
    features = pd.DataFrame(
        {"slot": index.hour * 2 + index.minute // 30, "dow": index.dayofweek, "month": index.month},
        index=index,
    )
    for k in LAG_WEEKS:
        features[f"lag_{k}w"] = y.reindex(index - pd.Timedelta(weeks=k)).to_numpy()
    features["lag_mean"] = features[[f"lag_{k}w" for k in LAG_WEEKS]].mean(axis=1)
    features["lag_52w"] = y.reindex(index - pd.Timedelta(weeks=52)).to_numpy()
    return features


def _frame(index: pd.DatetimeIndex, y: np.ndarray | None = None) -> pd.DataFrame:
    weekend = index.dayofweek >= 5
    frame = pd.DataFrame({"ds": index, "weekday": ~weekend, "weekend": weekend})
    if y is not None:
        frame["y"] = y
    return frame
