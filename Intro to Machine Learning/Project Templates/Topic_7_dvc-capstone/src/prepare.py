"""DVC pipeline stage 1 -- prepare (clean) the raw data.

A pipeline stage is just a plain function plus a tiny ``main()`` so DVC can run
it from the command line (see ``dvc.yaml``). Keeping the logic in an importable
function means the tests can check it directly, without DVC.

"Prepare" here means the small, safe cleaning every project needs before
modelling: drop exact duplicate rows, drop rows with missing values, and remove
impossible prices. The synthetic data is already tidy, so this mostly passes
through -- but the stage is where real-world messiness would be handled.

    python src/prepare.py        # or: dvc repro prepare
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import PREPARED_CSV, RAW_CSV  # noqa: E402  (import after sys.path fix)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw houses DataFrame."""
    cleaned = df.drop_duplicates().dropna()
    # A price must be positive to be a valid training target.
    cleaned = cleaned[cleaned["price"] > 0]
    return cleaned.reset_index(drop=True)


def main() -> None:
    """Read the raw CSV, clean it, and write the prepared CSV (stage output)."""
    df = pd.read_csv(RAW_CSV)
    prepared = prepare(df)
    PREPARED_CSV.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PREPARED_CSV, index=False)
    print(f"prepare: {len(df)} raw rows -> {len(prepared)} clean rows -> {PREPARED_CSV}")


if __name__ == "__main__":
    main()
