"""Generate a synthetic daily price history.

A forecasting template needs a time series. Downloading real market data (e.g.
with ``yfinance``) makes tests slow and flaky and depends on the network, so we
generate a *deterministic* synthetic series instead (fixed random seed). It has
a gentle upward trend plus mild seasonality and random noise -- enough to make
linear regression meaningful without being trivially perfect.

The output is committed to the repo, so you normally do NOT need to run this.
To use real data instead, replace ``generate`` with a ``yfinance`` download
that returns the same columns (``ticker``, ``date``, ``close``).

    python db/generate_raw.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import RAW_DIR  # noqa: E402  (import after sys.path fix)

# One synthetic ticker, a few years of trading days.
TICKER = "SYN"
N_DAYS = 500
SEED = 42
START_DATE = "2022-01-03"  # a Monday; we only keep weekdays below


def generate(n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    """Build a synthetic price series as trend + seasonality + noise."""
    rng = np.random.default_rng(seed)

    # ``t`` is a simple day index 0, 1, 2, ... -- the feature we regress on.
    t = np.arange(n_days)

    # A linear upward trend: price rises about 0.2 per day from a base of 100.
    # The trend is deliberately strong relative to the noise so a straight line
    # still explains most of the movement even on a short held-out window.
    trend = 100 + 0.2 * t

    # A slow seasonal wobble (period ~ one trading year of 252 days).
    seasonality = 1.5 * np.sin(2 * np.pi * t / 252)

    # Random day-to-day noise so the fit is not perfect.
    noise = rng.normal(0, 1.0, size=n_days)

    close = (trend + seasonality + noise).round(2)

    # Weekday-only dates so the series looks like real trading days.
    dates = pd.bdate_range(start=START_DATE, periods=n_days)

    return pd.DataFrame({"ticker": TICKER, "date": dates.strftime("%Y-%m-%d"), "close": close})


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "prices.csv"
    generate().to_csv(out, index=False)
    print(f"Wrote {out}")
