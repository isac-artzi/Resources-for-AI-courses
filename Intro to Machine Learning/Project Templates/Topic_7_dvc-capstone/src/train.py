"""DVC pipeline stage 3 -- train and compare baseline vs engineered.

We fit the *same* model (linear regression) on two feature sets and compare them
with 5-fold cross-validation, so the only thing that changes is the features.
That is the honest way to prove feature engineering helped. The stage then fits
the winning (engineered) model on all the data and saves it for the app's
"predict one house" page.

Metrics:
    * R^2  -- fraction of price variation explained (higher is better).
    * RMSE -- typical dollar error (lower is better).

    python src/train.py          # or: dvc repro train
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.featurize import (  # noqa: E402  (import after sys.path fix)
    TARGET,
    baseline_features,
    engineered_features,
)
from src.paths import (  # noqa: E402
    FEATURES_CSV,
    METRICS_JSON,
    MODEL_PATH,
    PREPARED_CSV,
)


def _cv_scores(X: pd.DataFrame, y: pd.Series) -> dict:
    """Cross-validate a linear model and return mean R^2 and RMSE."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
    neg_rmse = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    return {"r2": round(float(r2.mean()), 4), "rmse": round(float(-neg_rmse.mean()), 1)}


def compare(df: pd.DataFrame) -> dict:
    """Compare baseline vs engineered features via cross-validation."""
    y = df[TARGET]
    baseline = _cv_scores(baseline_features(df), y)
    engineered = _cv_scores(engineered_features(df), y)
    return {
        "baseline": baseline,
        "engineered": engineered,
        "r2_gain": round(engineered["r2"] - baseline["r2"], 4),
        "rmse_drop": round(baseline["rmse"] - engineered["rmse"], 1),
    }


def fit_final(df: pd.DataFrame) -> dict:
    """Fit the engineered model on all rows; return a saveable bundle.

    The bundle carries the training feature columns so a single-house prediction
    can be lined up to exactly the same columns.
    """
    X = engineered_features(df)
    model = LinearRegression().fit(X, df[TARGET])
    return {"model": model, "columns": list(X.columns)}


def predict_price(bundle: dict, house: pd.DataFrame) -> float:
    """Predict the price of one (or more) houses using a fitted bundle."""
    X = engineered_features(house).reindex(columns=bundle["columns"], fill_value=0)
    return float(bundle["model"].predict(X)[0])


def main() -> None:
    """Run the comparison, save metrics.json and the fitted model bundle.

    This stage consumes the featurize output (``features.csv``) for the
    engineered model and re-derives the baseline features from the prepared data
    for the side-by-side comparison.
    """
    engineered_df = pd.read_csv(FEATURES_CSV)
    prepared_df = pd.read_csv(PREPARED_CSV)

    y = engineered_df[TARGET]
    X_engineered = engineered_df.drop(columns=[TARGET])
    X_baseline = baseline_features(prepared_df)

    metrics = {
        "baseline": _cv_scores(X_baseline, y),
        "engineered": _cv_scores(X_engineered, y),
    }
    metrics["r2_gain"] = round(metrics["engineered"]["r2"] - metrics["baseline"]["r2"], 4)
    metrics["rmse_drop"] = round(metrics["baseline"]["rmse"] - metrics["engineered"]["rmse"], 1)
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2))

    model = LinearRegression().fit(X_engineered, y)
    bundle = {"model": model, "columns": list(X_engineered.columns)}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print(
        f"train: baseline R^2={metrics['baseline']['r2']}  "
        f"engineered R^2={metrics['engineered']['r2']}  "
        f"(+{metrics['r2_gain']})"
    )


if __name__ == "__main__":
    main()
