"""Mine "buy X -> buy Y" association rules from shopping baskets.

Association-rule mining answers questions like *"customers who buy toys also
tend to buy groceries -- how often, and how strongly?"* We use the classic
**Apriori** algorithm (from the ``mlxtend`` library) in two steps:

1. **Frequent itemsets** -- find sets of products that are bought together often
   enough (their *support*, the fraction of baskets containing them, clears a
   minimum).
2. **Rules** -- turn those itemsets into ``antecedent -> consequent`` rules and
   score each one:
     * **support**    -- how common the whole combination is.
     * **confidence** -- of the baskets with the antecedent, the fraction that
       also have the consequent (the strength of the "then" part).
     * **lift**       -- how much more likely the consequent is given the
       antecedent than by chance. Lift > 1 means a genuine positive association.

The input is the six ``bought_*`` columns, each a 0/1 flag, treated as a basket.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# The basket columns produced by db/generate_raw.py (one per product category).
BASKET_COLUMNS = [
    "bought_groceries",
    "bought_electronics",
    "bought_clothing",
    "bought_home",
    "bought_toys",
    "bought_beauty",
]


def baskets(df: pd.DataFrame) -> pd.DataFrame:
    """Return just the basket columns as booleans (what Apriori expects)."""
    return df[BASKET_COLUMNS].astype(bool)


def frequent_itemsets(df: pd.DataFrame, min_support: float = 0.2) -> pd.DataFrame:
    """Find product sets bought together in at least ``min_support`` of baskets."""
    return apriori(baskets(df), min_support=min_support, use_colnames=True)


def mine_rules(
    df: pd.DataFrame,
    min_support: float = 0.2,
    metric: str = "confidence",
    min_threshold: float = 0.6,
) -> pd.DataFrame:
    """Return association rules that clear ``min_threshold`` on ``metric``.

    Sorted by lift (strongest genuine associations first).
    """
    itemsets = frequent_itemsets(df, min_support=min_support)
    rules = association_rules(itemsets, metric=metric, min_threshold=min_threshold)
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def _format_itemset(items) -> str:
    """Turn a frozenset of column names into a readable, comma-joined string."""
    labels = sorted(name.removeprefix("bought_") for name in items)
    return ", ".join(labels)


def readable_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy, human-friendly view of the rules for display or testing.

    Converts the frozenset antecedents/consequents to plain text and keeps the
    key columns (support, confidence, lift).
    """
    if rules.empty:
        return pd.DataFrame(columns=["if_bought", "then_bought", "support", "confidence", "lift"])
    return pd.DataFrame(
        {
            "if_bought": rules["antecedents"].apply(_format_itemset),
            "then_bought": rules["consequents"].apply(_format_itemset),
            "support": rules["support"].round(3),
            "confidence": rules["confidence"].round(3),
            "lift": rules["lift"].round(3),
        }
    )
