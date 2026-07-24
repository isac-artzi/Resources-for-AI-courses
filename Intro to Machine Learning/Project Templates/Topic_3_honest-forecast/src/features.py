"""Turn a raw price table into features a model can learn from.

Linear regression needs numeric inputs (``X``) and a numeric target (``y``).
Our raw data has a date and a closing price. The simplest honest feature for a
trend model is a **day index**: 0 for the first day, 1 for the next, and so on.
The model then learns ``close ~ slope * t + intercept``.

Every function returns a NEW DataFrame and never mutates its input, so the
steps are easy to test and chain.
"""

import pandas as pd


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date, parse the date column, and add an integer day index ``t``.

    The day index is what we regress on. Sorting first guarantees ``t`` runs in
    calendar order, which matters for an honest time-ordered split later.
    """
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").reset_index(drop=True)
    # ``t`` = 0, 1, 2, ... in date order. This is the model's single feature.
    result["t"] = range(len(result))
    return result


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a prepared frame into features ``X`` (the day index) and target ``y``.

    scikit-learn expects ``X`` to be 2-D (a DataFrame of columns) and ``y`` to
    be 1-D (a Series), which is exactly what we return.
    """
    X = df[["t"]]
    y = df["close"]
    return X, y


def chronological_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series by TIME, not at random.

    This is the heart of an *honest* forecast. For ordinary data you shuffle
    before splitting, but for a time series that leaks the future into the
    training set. Instead we train on the earliest rows and test on the most
    recent ones -- exactly how a real forecast is judged.
    """
    df = prepare(df)
    cut = int(len(df) * (1 - test_frac))
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)
    return train, test
