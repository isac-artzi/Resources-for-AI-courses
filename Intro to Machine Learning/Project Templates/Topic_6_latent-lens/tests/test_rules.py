"""Tests for the association-rule mining in ``src/rules.py``."""

from src.rules import (
    BASKET_COLUMNS,
    baskets,
    frequent_itemsets,
    mine_rules,
    readable_rules,
)


def test_baskets_are_boolean(raw_shoppers):
    """The basket view should be boolean columns Apriori can consume."""
    b = baskets(raw_shoppers)
    assert list(b.columns) == BASKET_COLUMNS
    assert all(str(d) == "bool" for d in b.dtypes)


def test_frequent_itemsets_found(raw_shoppers):
    """Some product sets should clear the support threshold."""
    itemsets = frequent_itemsets(raw_shoppers, min_support=0.2)
    assert len(itemsets) > 0
    assert set(itemsets.columns) == {"support", "itemsets"}


def test_rules_are_sorted_by_lift(raw_shoppers):
    """Rules should be returned strongest-first (descending lift)."""
    rules = mine_rules(raw_shoppers, min_support=0.2, min_threshold=0.6)
    assert len(rules) > 0
    assert rules["lift"].is_monotonic_decreasing


def test_a_real_association_exists(raw_shoppers):
    """At least one rule should be a genuine association (lift > 1)."""
    rules = mine_rules(raw_shoppers, min_support=0.2, min_threshold=0.6)
    assert (rules["lift"] > 1.0).any()


def test_readable_rules_are_plain_text(raw_shoppers):
    """The readable view should replace frozensets with comma-joined strings."""
    rules = mine_rules(raw_shoppers, min_support=0.2, min_threshold=0.6)
    table = readable_rules(rules)
    assert list(table.columns) == [
        "if_bought",
        "then_bought",
        "support",
        "confidence",
        "lift",
    ]
    assert table["if_bought"].map(lambda s: isinstance(s, str)).all()
