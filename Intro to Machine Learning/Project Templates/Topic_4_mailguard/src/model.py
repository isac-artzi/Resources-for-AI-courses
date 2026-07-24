"""Train and evaluate a naive-Bayes spam filter, and explain its decisions.

The pipeline has two stages, both from scikit-learn:

    1. ``TfidfVectorizer`` turns each message into a vector of word weights.
    2. ``MultinomialNB`` (naive Bayes) learns how spammy each word is and
       multiplies the evidence together to score a new message.

We wrap them in a single ``Pipeline`` so the exact same text preprocessing is
applied during training and prediction -- a common source of bugs when done by
hand. ``alpha=1.0`` is Laplace (add-one) smoothing, which stops a single unseen
word from forcing a probability of exactly zero.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# We treat "spam" as the positive class (label 1) throughout.
SPAM = "spam"
HAM = "ham"


def make_pipeline() -> Pipeline:
    """Build the (unfitted) text -> tf-idf -> naive-Bayes pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True)),
            ("nb", MultinomialNB(alpha=1.0)),
        ]
    )


def split(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    """Split into train/test, keeping the spam/ham balance (``stratify``).

    A fixed ``random_state`` makes the split reproducible so everyone sees the
    same numbers.
    """
    X = df["text"]
    y = (df["label"] == SPAM).astype(int)  # 1 = spam, 0 = ham
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train(X_train, y_train) -> Pipeline:
    """Fit the pipeline on the training messages."""
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    return pipe


def evaluate(pipe: Pipeline, X_test, y_test) -> dict:
    """Score the filter with the standard classification metrics.

    * **precision** -- of the mail we flagged as spam, how much really was?
    * **recall**    -- of all real spam, how much did we catch?
    * **F1**        -- the balance between precision and recall.
    * **ROC-AUC**   -- ranking quality across all thresholds.
    """
    predictions = pipe.predict(X_test)
    spam_prob = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, spam_prob)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def classify_message(pipe: Pipeline, text: str) -> float:
    """Return the probability that a single message is spam (0..1)."""
    return float(pipe.predict_proba([text])[0, 1])


def influential_words(pipe: Pipeline, top_n: int = 5) -> pd.DataFrame:
    """List the words that push most strongly toward spam.

    Naive Bayes stores, for each word, its log-probability under each class.
    The difference (spam minus ham) is how much seeing that word tips the scale
    toward spam. We return the ``top_n`` most spam-indicative words.
    """
    vectorizer: TfidfVectorizer = pipe.named_steps["tfidf"]
    nb: MultinomialNB = pipe.named_steps["nb"]
    words = np.array(vectorizer.get_feature_names_out())
    # Row 0 = ham (class 0), row 1 = spam (class 1).
    spam_minus_ham = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    order = np.argsort(spam_minus_ham)[::-1][:top_n]
    return pd.DataFrame({"word": words[order], "spam_evidence": spam_minus_ham[order].round(3)})
