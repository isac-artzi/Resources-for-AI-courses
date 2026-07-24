"""Describe what *good* data looks like, then check data against it.

A schema is a written contract: every column, its type, and the range of
values we accept. ``pandera`` turns that contract into runnable code. The raw
data should FAIL this schema (that is the whole point of the topic); the
cleaned data should PASS it.

We keep the rules in one place so the UI, the tests, and future teammates all
agree on the definition of "clean".
"""

import pandera as pa

from src.clean import MAX_AGE, MIN_AGE, VALID_COUNTRIES, VALID_PLANS

# The contract. Each Column says: what type, and what values are allowed.
# ``coerce=True`` lets pandera cast a column to the declared type before
# checking it, so a whole-number float like 40.0 is accepted as an int.
SIGNUPS_SCHEMA = pa.DataFrameSchema(
    {
        "signup_id": pa.Column(int, unique=True),
        "name": pa.Column(str, nullable=False),
        "age": pa.Column(int, pa.Check.in_range(MIN_AGE, MAX_AGE), coerce=True),
        "country": pa.Column(str, pa.Check.isin(VALID_COUNTRIES)),
        "signup_date": pa.Column("datetime64[ns]", coerce=True),
        "income": pa.Column(float, pa.Check.ge(0), coerce=True),
        "plan": pa.Column(str, pa.Check.isin(VALID_PLANS)),
    }
)


def validate(df) -> tuple[bool, "pd.DataFrame | None"]:  # noqa: F821
    """Check ``df`` against the schema.

    Returns ``(True, None)`` when the data passes. When it fails we return
    ``(False, failure_cases)`` -- a DataFrame with one row per broken rule
    (which column, which check, and the offending value) so the UI can show a
    readable report instead of crashing. ``lazy=True`` collects *all* failures
    in one pass rather than stopping at the first one.
    """
    try:
        SIGNUPS_SCHEMA.validate(df, lazy=True)
        return True, None
    except pa.errors.SchemaErrors as exc:
        return False, exc.failure_cases
