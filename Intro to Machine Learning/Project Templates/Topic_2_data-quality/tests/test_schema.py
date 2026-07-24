"""Tests for the pandera schema in ``src/schema.py``.

The core promise of this topic: the RAW data fails the schema, and the CLEANED
data passes it. These two tests pin that promise down.
"""

from src.clean import run_cleaning
from src.schema import validate


def test_raw_data_fails_validation(raw_signups):
    ok, failures = validate(raw_signups)
    assert ok is False
    # We should get a report with at least one broken rule.
    assert failures is not None
    assert len(failures) > 0


def test_cleaned_data_passes_validation(raw_signups):
    cleaned = run_cleaning(raw_signups)
    ok, failures = validate(cleaned)
    assert ok is True
    assert failures is None
