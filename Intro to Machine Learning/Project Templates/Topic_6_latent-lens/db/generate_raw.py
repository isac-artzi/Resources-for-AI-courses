"""Generate a synthetic shopper dataset for unsupervised learning.

This topic is about finding structure in data *without* labels: grouping similar
customers with **k-means**, summarising many features into two with **PCA**, and
mining "customers who buy X also buy Y" patterns with **association rules**.

To make those techniques show something clear, we build the data with a hidden
structure and then let the algorithms rediscover it:

* Every shopper secretly belongs to one of three **segments** (budget, family,
  premium). Each segment has its own typical age, income, spending score, visit
  frequency, and basket value -- so the numeric features form three clusters.
* Each segment also has its own **shopping basket** habits (which product
  categories they tend to buy), so association-rule mining finds real patterns
  such as "toys -> groceries" for families.

The ``segment`` column is the ground truth. Clustering never sees it -- we keep
it only to colour the plots and to check, in a test, that k-means recovers the
three groups. Everything is generated *deterministically* (fixed seed) so the
committed CSV and the tests are reproducible.

    python db/generate_raw.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import RAW_DIR  # noqa: E402  (import after sys.path fix)

N_PER_SEGMENT = 300
SEED = 42

# The six product categories that make up a shopping basket.
CATEGORIES = ["groceries", "electronics", "clothing", "home", "toys", "beauty"]

# Each segment is a recipe: typical numeric feature values (mean, spread) plus
# the probability of buying each product category. The numeric means are far
# enough apart that the segments form three distinct clouds in feature space.
SEGMENTS = {
    0: {
        "name": "Budget-Conscious",
        # (mean, std) for each numeric feature.
        "age": (28, 5),
        "annual_income": (30_000, 6_000),
        "spending_score": (35, 8),
        "visits_per_month": (2.0, 0.8),
        "avg_basket_value": (25, 8),
        # P(buys) for each product category, in CATEGORIES order.
        "buys": [0.60, 0.10, 0.75, 0.10, 0.10, 0.55],
    },
    1: {
        "name": "Family",
        "age": (42, 6),
        "annual_income": (65_000, 9_000),
        "spending_score": (55, 8),
        "visits_per_month": (6.0, 1.2),
        "avg_basket_value": (90, 15),
        "buys": [0.85, 0.30, 0.40, 0.40, 0.85, 0.30],
    },
    2: {
        "name": "Premium",
        "age": (50, 7),
        "annual_income": (110_000, 15_000),
        "spending_score": (80, 9),
        "visits_per_month": (4.0, 1.0),
        "avg_basket_value": (180, 30),
        "buys": [0.50, 0.85, 0.40, 0.80, 0.20, 0.60],
    },
}


def generate(n_per_segment: int = N_PER_SEGMENT, seed: int = SEED) -> pd.DataFrame:
    """Build the shopper DataFrame: three segments stacked together."""
    rng = np.random.default_rng(seed)
    frames = []

    for segment_id, recipe in SEGMENTS.items():
        n = n_per_segment
        # Numeric features: one normal draw per feature, clipped to stay sensible.
        age = rng.normal(*recipe["age"], size=n).clip(18, 90).round().astype(int)
        income = (
            rng.normal(*recipe["annual_income"], size=n).clip(10_000, 250_000).round().astype(int)
        )
        spending = rng.normal(*recipe["spending_score"], size=n).clip(1, 100).round().astype(int)
        visits = rng.normal(*recipe["visits_per_month"], size=n).clip(0, 30).round(1)
        basket = rng.normal(*recipe["avg_basket_value"], size=n).clip(5, 500).round(2)

        data = {
            "age": age,
            "annual_income": income,
            "spending_score": spending,
            "visits_per_month": visits,
            "avg_basket_value": basket,
        }
        # Basket columns: a coin flip per category using this segment's odds.
        for category, prob in zip(CATEGORIES, recipe["buys"]):
            data[f"bought_{category}"] = (rng.random(size=n) < prob).astype(int)

        data["segment"] = segment_id
        frames.append(pd.DataFrame(data))

    # Stack the three segments and shuffle so the rows are not grouped by segment.
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "shoppers.csv"
    generate().to_csv(out, index=False)
    print(f"Wrote {out}")
