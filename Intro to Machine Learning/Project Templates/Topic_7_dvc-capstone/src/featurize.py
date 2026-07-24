"""DVC pipeline stage 2 -- build model inputs (feature engineering).

This is the heart of the capstone. We offer two feature builders on the *same*
cleaned data so we can measure whether engineering actually helps:

* ``baseline_features`` -- the naive version: raw numeric columns, and the
  neighborhood turned into an arbitrary integer code. No transforms.
* ``engineered_features`` -- four classic techniques:
    1. **One-hot encoding** of ``neighborhood`` (so each area gets its own
       weight instead of a meaningless number line).
    2. **Log transform** (``log1p``) of the right-skewed ``sqft`` and
       ``lot_size`` (pulls in the long tail so a linear model fits better).
    3. **Binning** of house age into eras (a non-linear effect made explicit).
    4. **Ratio features** ``bed_bath_ratio`` and ``sqft_per_bedroom`` (combine
       columns into something more informative than either alone).

Both builders are *stateless* and work on a single row too, so the same code
powers the "predict one house" page.

    python src/featurize.py      # or: dvc repro featurize
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import FEATURES_CSV, PREPARED_CSV  # noqa: E402  (import after sys.path fix)

TARGET = "price"

# Raw numeric columns used as-is by the baseline model.
RAW_NUMERIC = [
    "sqft",
    "lot_size",
    "year_built",
    "bedrooms",
    "bathrooms",
    "garage",
    "condition",
]

# We compute house age relative to this year, then bin it into fixed "eras".
CURRENT_YEAR = 2024
AGE_BIN_EDGES = [-1, 10, 25, 50, np.inf]
AGE_BIN_LABELS = ["0-10 yrs", "10-25 yrs", "25-50 yrs", "50+ yrs"]
# The five neighborhoods, listed so one-hot columns are stable even for a single
# house that only contains one of them.
NEIGHBORHOODS = ["Ashby", "Brook", "Cedar", "Delta", "Echo"]


def baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Naive features: raw numbers plus a label-encoded neighborhood."""
    X = df[RAW_NUMERIC].copy()
    # Turn the category into an arbitrary integer code (A=0, B=1, ...). This is
    # the common beginner mistake we want to out-perform.
    codes = pd.Categorical(df["neighborhood"], categories=NEIGHBORHOODS).codes
    X["neighborhood_code"] = codes
    return X


def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features: one-hot + log transforms + age bins + ratios."""
    # 1. One-hot encode neighborhood against the fixed category list.
    neighborhood = pd.Categorical(df["neighborhood"], categories=NEIGHBORHOODS)
    onehot = pd.get_dummies(neighborhood, prefix="nb")
    onehot.index = df.index

    # Numeric columns that carry over unchanged.
    base = df[["bedrooms", "bathrooms", "garage", "condition"]].copy()

    # 2. Log-transform the skewed size features.
    base["log_sqft"] = np.log1p(df["sqft"])
    base["log_lot_size"] = np.log1p(df["lot_size"])

    # 3. Bin house age into fixed eras, then one-hot encode.
    house_age = CURRENT_YEAR - df["year_built"]
    era = pd.cut(house_age, bins=AGE_BIN_EDGES, labels=AGE_BIN_LABELS)
    era = pd.Categorical(era, categories=AGE_BIN_LABELS)
    age_onehot = pd.get_dummies(era, prefix="age")
    age_onehot.index = df.index

    # 4. Ratio features combining columns.
    base["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"]
    base["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"]

    return pd.concat([onehot, age_onehot, base], axis=1)


def main() -> None:
    """Write the engineered features (plus the target) as the stage output."""
    df = pd.read_csv(PREPARED_CSV)
    engineered = engineered_features(df)
    engineered[TARGET] = df[TARGET].values
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_csv(FEATURES_CSV, index=False)
    print(f"featurize: {engineered.shape[1] - 1} engineered columns -> {FEATURES_CSV}")


if __name__ == "__main__":
    main()
