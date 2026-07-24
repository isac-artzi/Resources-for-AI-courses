"""Generate a deliberately messy synthetic dataset.

Real data is never clean, so a data-quality template needs data with real
problems: missing values, duplicates, outliers, and inconsistent text. This
script builds such a dataset *deterministically* (fixed random seed) so the
tests are reproducible, and writes it to ``data/raw/signups.csv``.

The output is committed to the repo, so you normally do NOT need to run this.
It is here so you can see exactly how the messiness was introduced.

    python db/generate_raw.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import RAW_DIR  # noqa: E402  (import after sys.path fix)

N_ROWS = 400
SEED = 42


def generate(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Build the messy DataFrame. See the inline comments for each problem."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "signup_id": np.arange(1, n_rows + 1),
            "name": [f"User {i}" for i in range(1, n_rows + 1)],
            # Ages centered around 40; we will corrupt some below.
            "age": rng.integers(18, 71, size=n_rows).astype(float),
            # Inconsistent country spellings/casing/whitespace on purpose.
            "country": rng.choice(
                ["USA", "usa ", " Canada", "canada", "Brazil", "BRAZIL", "India"],
                size=n_rows,
            ),
            "signup_date": pd.to_datetime("2023-01-01")
            + pd.to_timedelta(rng.integers(0, 365, size=n_rows), unit="D"),
            # Income as a float; some will be corrupted to negatives.
            "income": rng.normal(70000, 25000, size=n_rows).round(2),
            # Plan names with inconsistent casing.
            "plan": rng.choice(["free", "Free", "pro", "PRO", "enterprise"], size=n_rows),
        }
    )
    # Dates are stored as text in the raw file (as they would be from a CSV).
    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")

    # --- Inject quality problems -------------------------------------------
    # 1. Missing values: blank out ~5% of ages and incomes.
    age_nulls = rng.choice(n_rows, size=n_rows // 20, replace=False)
    df.loc[age_nulls, "age"] = np.nan
    income_nulls = rng.choice(n_rows, size=n_rows // 20, replace=False)
    df.loc[income_nulls, "income"] = np.nan

    # 2. Outliers: a few impossible ages and negative incomes.
    df.loc[rng.choice(n_rows, size=4, replace=False), "age"] = 999
    df.loc[rng.choice(n_rows, size=4, replace=False), "income"] = -5000.0

    # 3. Duplicates: copy 15 random rows and append them.
    dupes = df.sample(15, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    return df


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "signups.csv"
    generate().to_csv(out, index=False)
    print(f"Wrote {out}")
