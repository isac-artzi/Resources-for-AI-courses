"""Train and evaluate a linear-regression forecaster, and project it forward.

This module keeps the machine-learning code in one small place:
    * ``train_linear``   -- fit the model on the training rows.
    * ``evaluate``       -- score it on the held-out (future) rows.
    * ``residual_std``   -- measure typical error, for an honest forecast band.
    * ``forecast``       -- project the trend beyond the last observed day.

We use scikit-learn's ``LinearRegression`` -- ordinary least squares, the
workhorse first model of any ML course.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.features import make_xy


def train_linear(train_df: pd.DataFrame) -> LinearRegression:
    """Fit ``close ~ t`` on the training rows and return the fitted model."""
    X, y = make_xy(train_df)
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate(model: LinearRegression, test_df: pd.DataFrame) -> dict:
    """Score the model on held-out rows with R^2 and RMSE.

    * **R^2** answers "what fraction of the variation did we explain?" (1.0 is
      perfect, 0.0 is no better than always guessing the mean).
    * **RMSE** is the typical error in the same units as the price (dollars).
    """
    X, y = make_xy(test_df)
    predictions = model.predict(X)
    errors = y.to_numpy() - predictions
    rmse = float(np.sqrt(np.mean(errors**2)))
    r2 = float(model.score(X, y))
    return {"r2": r2, "rmse": rmse}


def residual_std(model: LinearRegression, train_df: pd.DataFrame) -> float:
    """Standard deviation of the training residuals (typical miss size).

    We use this to draw an honest uncertainty band around the forecast: a
    point estimate alone pretends we know the future exactly, which we do not.
    """
    X, y = make_xy(train_df)
    residuals = y.to_numpy() - model.predict(X)
    return float(np.std(residuals))


def forecast(
    model: LinearRegression,
    history: pd.DataFrame,
    horizon: int,
    resid_std: float,
) -> pd.DataFrame:
    """Project the trend ``horizon`` business days past the last observed day.

    Returns a DataFrame with the future ``date``, the ``predicted`` price, and
    a ``lower``/``upper`` band of +/- two residual standard deviations (a rough
    95% interval for a normal error).
    """
    last_t = int(history["t"].max())
    last_date = pd.to_datetime(history["date"]).max()

    future_t = np.arange(last_t + 1, last_t + 1 + horizon)
    future_dates = pd.bdate_range(start=last_date, periods=horizon + 1)[1:]

    predicted = model.predict(pd.DataFrame({"t": future_t}))
    band = 2 * resid_std
    return pd.DataFrame(
        {
            "date": future_dates,
            "predicted": predicted.round(2),
            "lower": (predicted - band).round(2),
            "upper": (predicted + band).round(2),
        }
    )
