"""Generate a synthetic house-price dataset for a feature-engineering capstone.

This topic is about two professional habits:

* **Feature engineering** -- turning raw columns into better inputs (encoding
  categories, taming skew with logs, binning, building ratios) and *measuring*
  whether it actually helps.
* **Reproducible pipelines** -- wiring the steps together with DVC so anyone can
  rebuild the exact result from the raw data.

To make feature engineering clearly pay off, the data is generated so that a
naive model struggles and a well-engineered one wins:

* ``neighborhood`` has a strong effect on price, but the premiums are **not** in
  alphabetical order. A model that treats the category as a number (A=0, B=1,
  ...) gets a meaningless ranking; **one-hot encoding** recovers the real
  premiums.
* ``sqft`` and ``lot_size`` are **right-skewed** (a few very large houses), so a
  ``log1p`` transform helps a linear model.

Everything is generated *deterministically* (fixed seed) so the committed CSV
and the tests are reproducible.

    python db/generate_raw.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import RAW_DIR  # noqa: E402  (import after sys.path fix)

N_ROWS = 1500
SEED = 42
CURRENT_YEAR = 2024

# Neighborhood -> (share of houses, price premium in dollars). The premiums are
# deliberately NOT in alphabetical order, so label-encoding the category as a
# number is misleading and one-hot encoding is the right tool.
NEIGHBORHOODS = {
    "Ashby": (0.30, 220_000),
    "Brook": (0.15, 400_000),
    "Cedar": (0.10, 120_000),
    "Delta": (0.20, 300_000),
    "Echo": (0.25, 160_000),
}


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Build the house DataFrame from raw, partly-skewed features."""
    rng = np.random.default_rng(seed)

    names = list(NEIGHBORHOODS)
    shares = np.array([NEIGHBORHOODS[n][0] for n in names])
    neighborhood = rng.choice(names, size=n_rows, p=shares / shares.sum())
    premium = np.array([NEIGHBORHOODS[n][1] for n in neighborhood])

    # Right-skewed sizes: most houses modest, a few very large (lognormal).
    sqft = rng.lognormal(mean=np.log(1700), sigma=0.35, size=n_rows)
    sqft = sqft.clip(600, 6000).round().astype(int)
    lot_size = rng.lognormal(mean=np.log(6000), sigma=0.6, size=n_rows)
    lot_size = lot_size.clip(1000, 60000).round().astype(int)

    year_built = rng.integers(1950, CURRENT_YEAR, size=n_rows)
    house_age = CURRENT_YEAR - year_built
    bedrooms = rng.integers(1, 7, size=n_rows)
    bathrooms = rng.integers(1, 5, size=n_rows)
    garage = (rng.random(size=n_rows) < 0.6).astype(int)
    condition = rng.integers(1, 6, size=n_rows)

    # Hidden pricing rule (what the model tries to learn) plus noise.
    price = (
        premium
        + 110 * sqft
        + 0.8 * lot_size
        + 12_000 * bedrooms
        + 18_000 * bathrooms
        + 22_000 * garage
        + 16_000 * condition
        - 900 * house_age
        + rng.normal(0, 25_000, size=n_rows)
    )
    price = price.clip(30_000).round().astype(int)

    return pd.DataFrame(
        {
            "neighborhood": neighborhood,
            "sqft": sqft,
            "lot_size": lot_size,
            "year_built": year_built,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "garage": garage,
            "condition": condition,
            "price": price,
        }
    )


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "houses.csv"
    generate().to_csv(out, index=False)
    print(f"Wrote {out}")
