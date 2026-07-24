"""Generate a synthetic customer-churn dataset for classification.

A classifier-comparison template needs a tabular dataset with a couple of
classes and features that genuinely predict the label. We build one
*deterministically* (fixed random seed) so the tests are reproducible.

The task: predict whether a customer **churned** (left the service). The label
is produced by a hidden logistic rule -- churn is more likely for customers who
make many support calls, report low satisfaction, pay high monthly charges, and
have short tenure -- plus random noise so no model can be perfect.

The output is committed to the repo, so you normally do NOT need to run this.
To use a real dataset instead, replace ``generate`` with a loader that returns
the same feature columns plus a 0/1 ``churned`` column.

    python db/generate_raw.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import RAW_DIR  # noqa: E402  (import after sys.path fix)

N_ROWS = 2000
SEED = 42


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Build the churn DataFrame from interpretable numeric features."""
    rng = np.random.default_rng(seed)

    # Interpretable customer features on realistic scales.
    tenure_months = rng.integers(1, 73, size=n_rows)          # 1..72 months
    monthly_charges = rng.normal(70, 25, size=n_rows).clip(15, 200).round(2)
    support_calls = rng.poisson(2, size=n_rows)               # count of calls
    age = rng.integers(18, 81, size=n_rows)                   # 18..80 years
    num_services = rng.integers(1, 7, size=n_rows)            # 1..6 add-ons
    satisfaction = rng.integers(1, 11, size=n_rows)           # 1..10 survey score

    # Hidden logistic rule. We standardize each feature roughly by hand so the
    # weights are comparable, then combine them into a churn "score".
    score = (
        0.9 * (support_calls - 2) / 2      # more calls -> more churn
        - 0.9 * (satisfaction - 5) / 3     # happier    -> less churn
        + 0.6 * (monthly_charges - 70) / 25  # pricier -> more churn
        - 0.7 * (tenure_months - 36) / 20  # loyal      -> less churn
        - 0.2 * (num_services - 3) / 2     # more tied-in -> slightly less churn
    )
    # Convert the score to a probability, add a little noise, and draw the
    # label. The noise keeps the task realistic (no model reaches 100%) without
    # drowning out the signal.
    prob = 1 / (1 + np.exp(-score))
    noise = rng.normal(0, 0.12, size=n_rows)
    churned = ((prob + noise) > 0.5).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "age": age,
            "num_services": num_services,
            "satisfaction": satisfaction,
            "churned": churned,
        }
    )


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "customers.csv"
    generate().to_csv(out, index=False)
    print(f"Wrote {out}")
