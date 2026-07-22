"""Synthetic tabular dataset generator (Adult-Income shaped).

Lives in ``shared`` so that BOTH the API tier (to train) and the db/seed script
(to persist a default dataset) import the exact same generating function. The
data mimics the UCI *Adult Income* schema (a few numeric + categorical features,
a binary >50K target) but is synthesized from a KNOWN logistic rule so the
numerical pytest test can assert the classifier recovers that signal.

No real PII is ever stored -- every row is fabricated.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# The feature contract shared across all three clouds. The API's sklearn
# ColumnTransformer and the UI's Predict form both read these lists.
NUMERIC_COLS: List[str] = ["age", "education_num", "hours_per_week", "capital_gain"]
CATEGORICAL_COLS: List[str] = ["workclass", "marital_status", "occupation"]
FEATURE_COLS: List[str] = NUMERIC_COLS + CATEGORICAL_COLS

WORKCLASS: List[str] = ["Private", "Self-emp", "Government", "Other"]
MARITAL: List[str] = ["Married", "Never-married", "Divorced"]
OCCUPATION: List[str] = ["Exec", "Prof", "Sales", "Admin", "Craft", "Service"]

CATEGORIES: Dict[str, List[str]] = {
    "workclass": WORKCLASS,
    "marital_status": MARITAL,
    "occupation": OCCUPATION,
}

TARGET_NAME = "income"
TARGET_CLASSES = ["<=50K", ">50K"]  # index 0 / 1

# Ground-truth offsets that define the (synthetic) generating rule.
_WORKCLASS_OFFSET = {"Private": 0.0, "Self-emp": 0.2, "Government": 0.1, "Other": -0.3}
_MARITAL_OFFSET = {"Married": 0.8, "Never-married": -0.6, "Divorced": -0.2}
_OCC_OFFSET = {
    "Exec": 0.8, "Prof": 0.7, "Sales": 0.1, "Admin": 0.0, "Craft": -0.1, "Service": -0.5,
}


def generate_tabular(
    n_rows: int,
    noise: float = 1.0,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], List[int]]:
    """Return ``(records, labels)`` for a synthetic income-classification set.

    ``records`` is a list of dicts keyed by :data:`FEATURE_COLS`; ``labels`` is a
    list of 0/1 ints (1 == ">50K"). A fixed ``seed`` makes the dataset
    reproducible so run comparisons in the Supabase ``runs`` table are
    apples-to-apples. ``noise`` scales a small amount of label flipping so the
    task is not perfectly separable (5% flips at ``noise=1.0``).
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(17, 76, n_rows)
    education_num = rng.integers(1, 17, n_rows)
    hours = rng.integers(20, 61, n_rows)
    has_gain = rng.random(n_rows) < 0.15
    capital_gain = (has_gain * rng.integers(1000, 15001, n_rows)).astype(int)

    workclass = rng.choice(WORKCLASS, n_rows, p=[0.60, 0.15, 0.20, 0.05])
    marital = rng.choice(MARITAL, n_rows, p=[0.50, 0.35, 0.15])
    occupation = rng.choice(OCCUPATION, n_rows)

    z = (
        0.04 * (age - 38)
        + 0.25 * (education_num - 10)
        + 0.03 * (hours - 40)
        + 0.00008 * capital_gain
        + np.array([_WORKCLASS_OFFSET[w] for w in workclass])
        + np.array([_MARITAL_OFFSET[m] for m in marital])
        + np.array([_OCC_OFFSET[o] for o in occupation])
        - 0.5
    )
    base = (z >= 0).astype(int)

    flip_p = 0.05 * float(noise)
    flips = rng.random(n_rows) < flip_p
    labels = np.where(flips, 1 - base, base).astype(int)

    records = [
        {
            "age": int(age[i]),
            "education_num": int(education_num[i]),
            "hours_per_week": int(hours[i]),
            "capital_gain": int(capital_gain[i]),
            "workclass": str(workclass[i]),
            "marital_status": str(marital[i]),
            "occupation": str(occupation[i]),
        }
        for i in range(n_rows)
    ]
    return records, labels.tolist()
