"""Build, train, and compare three classic classifiers.

The whole point of this topic is a fair, apples-to-apples comparison. So every
model is wrapped in the SAME scikit-learn ``Pipeline`` (StandardScaler -> the
classifier), trained on the SAME stratified split with the SAME random seed, and
scored with the SAME metrics. Only the classifier changes.

Why scale first? k-nearest-neighbours and the RBF SVM both measure distances
between points, so a feature on a big scale (like income) would dominate a
feature on a small scale (like a 1-10 score) unless we standardize them. The
decision tree does not need scaling, but keeping it in the pipeline makes the
comparison uniform and does no harm.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

TARGET = "churned"
# The six numeric features the models learn from.
FEATURES = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "age",
    "num_services",
    "satisfaction",
]


def make_models() -> dict[str, Pipeline]:
    """Return the three (unfitted) model pipelines, keyed by name.

    Each is ``StandardScaler`` followed by one classifier with sensible,
    beginner-friendly defaults.
    """
    return {
        "kNN": Pipeline(
            [("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ]
        ),
    }


def split(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    """Stratified, reproducible train/test split of features and target."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    """Score a fitted model with the standard classification metrics."""
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def train_and_compare(df: pd.DataFrame) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """Fit all three models on the same split and return them plus a scoreboard.

    Returns ``(fitted_models, comparison_df)`` where ``comparison_df`` has one
    row per model and one column per metric, sorted by ROC-AUC (best first).
    """
    X_train, X_test, y_train, y_test = split(df)
    fitted = {}
    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        scores = evaluate(model, X_test, y_test)
        scores.pop("confusion_matrix")  # keep the table numeric and tidy
        rows.append({"model": name, **{k: round(v, 3) for k, v in scores.items()}})
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return fitted, comparison


def recommend(comparison: pd.DataFrame) -> str:
    """Return the name of the best model (highest ROC-AUC)."""
    return comparison.iloc[0]["model"]
