"""Seasonal Naive, MSTL and Prophet behind one interface.

Each model fits on the history it is given and forecasts the `horizon` slots that follow
it. Settings are fixed in advance; none is tuned on the test window.
"""

from typing import Protocol

import numpy as np
import pandas as pd

WEEK = 336


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


def _frame(index: pd.DatetimeIndex, y: np.ndarray | None = None) -> pd.DataFrame:
    weekend = index.dayofweek >= 5
    frame = pd.DataFrame({"ds": index, "weekday": ~weekend, "weekend": weekend})
    if y is not None:
        frame["y"] = y
    return frame
