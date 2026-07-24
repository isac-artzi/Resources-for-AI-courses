"""Tests for training, evaluation, and forecasting in ``src/model.py``.

We use the real synthetic series (a genuine trend), so we can assert the model
actually fits it well and that the forecast has a sensible shape.
"""

from src.features import chronological_split, prepare
from src.model import evaluate, forecast, residual_std, train_linear


def test_model_fits_the_trend(raw_prices):
    train, test = chronological_split(raw_prices, test_frac=0.2)
    model = train_linear(train)
    scores = evaluate(model, test)
    # A clean synthetic upward trend should be explained well by a line.
    assert scores["r2"] > 0.8
    assert scores["rmse"] > 0  # some error always remains (there is noise)


def test_learned_slope_is_positive(raw_prices):
    model = train_linear(prepare(raw_prices))
    # Our synthetic data trends upward, so the slope must be positive.
    assert float(model.coef_[0]) > 0


def test_forecast_shape_and_band(raw_prices):
    history = prepare(raw_prices)
    model = train_linear(history)
    resid = residual_std(model, history)
    future = forecast(model, history, horizon=10, resid_std=resid)

    assert len(future) == 10
    assert list(future.columns) == ["date", "predicted", "lower", "upper"]
    # The band must bracket the point estimate on every row.
    assert (future["lower"] <= future["predicted"]).all()
    assert (future["predicted"] <= future["upper"]).all()
    # The forecast starts after the last observed date.
    assert future["date"].min() > history["date"].max()
