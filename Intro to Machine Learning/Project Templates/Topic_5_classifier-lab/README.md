# Topic 5 — Classification: ClassifierLab

> **Live app:** _paste your Streamlit Community Cloud URL here after deploying_
> **Repository:** _paste your GitHub repository URL here_

A starter **product** template for an introductory machine-learning course.
It is a small, deployed Streamlit application that trains and **fairly compares
three classic classifiers** — k-nearest-neighbours, an RBF support-vector
machine, and a decision tree — then recommends one. It reuses the **universal
build pattern** from Topic 1:

```
raw CSV  ->  SQLite (basic CRUD)  ->  pandas + sklearn  ->  Streamlit UI
```

This README is written as **product documentation**, using the standard report
sections: Problem, Theory, Data, Method, Results, Ethics, References.

---

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.11+).
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies.
pip install -r requirements-dev.txt

# 3. Build the SQLite database from the raw CSV (basic CRUD).
python db/build_sqlite.py

# 4. Run the app locally.
streamlit run app.py

# 5. (Optional) run the tests and linter, exactly as CI does.
pytest -q
ruff check .
```

> You do not strictly need step 3 by hand: the app builds the database
> automatically on first run. The script is there so you can see the CRUD.

---

## Problem

Given a labelled dataset, which classifier should you actually deploy? Beginners
often try one model, see a decent number, and stop. The goal here is a
disciplined, **apples-to-apples comparison** that changes only the classifier —
holding the data, split, seed, and preprocessing constant — and then makes a
clear recommendation.

## Theory

**k-nearest-neighbours** classifies a point by the majority vote of its closest
neighbours. **Support-vector machines** (RBF kernel) find a curved boundary that
separates the classes with the widest margin. **Decision trees** ask a sequence
of threshold questions; capped at `max_depth=5` they stay readable.

**Why scale first?** kNN and the RBF SVM measure distances between points, so a
feature on a large scale would dominate one on a small scale. Wrapping each model
in a `Pipeline` with `StandardScaler` fixes this — and applies the exact same
scaling at prediction time.

**Fair comparison** means one stratified split (`stratify=y`), one fixed
`random_state=42`, and the same metrics for every model: accuracy, precision,
recall, F1, and ROC-AUC. We rank by ROC-AUC (threshold-independent ranking
quality) to pick a winner.

## Data

A **synthetic** 2000-row customer-churn dataset (`data/raw/customers.csv`) with
six interpretable numeric features and a binary `churned` target, generated
deterministically by `db/generate_raw.py`. See
[`data/raw/README.md`](data/raw/README.md).

## Method

1. **Load** — `db/build_sqlite.py` reads the CSV into SQLite (`CREATE` +
   `INSERT`) plus a `provenance` table.
2. **Read** — `src/db.py` reads the table into a pandas DataFrame.
3. **Split** — `src/model.py` makes one stratified, seeded train/test split.
4. **Train** — three `StandardScaler` + classifier pipelines are fitted on the
   same training data.
5. **Compare** — each is scored on the same test set; a scoreboard ranks them by
   ROC-AUC and `recommend` names the winner.
6. **Predict** — the app scores a user-entered customer with all three models.

## Results

The deployed app has three pages: **Dataset** (features, class balance, and how
each feature relates to churn), **Compare Models** (a scoreboard, per-model
confusion matrices, and a recommendation), and **Try a Prediction** (enter a
customer and compare the three models' outputs).

## Ethics

A churn model can drive real decisions — who gets a retention discount, who gets
ignored. Optimising a single metric can hide unequal error rates across groups
(for example, by age). Report precision and recall, not just accuracy, and check
whether the "best" model is best for everyone or only on average. A slightly
weaker but interpretable decision tree may be preferable when a decision must be
explained to the person it affects. _(Replace this section with your own
~300-word reflection when you adapt the template.)_

## References

- scikit-learn — nearest neighbours: https://scikit-learn.org/stable/modules/neighbors.html
- scikit-learn — support vector machines: https://scikit-learn.org/stable/modules/svm.html
- scikit-learn — decision trees: https://scikit-learn.org/stable/modules/tree.html
- scikit-learn — `Pipeline`: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

---

See [`../Topic_1_sql-foundations/TUTORIAL.md`](../Topic_1_sql-foundations/TUTORIAL.md)
for the full, step-by-step build-and-deploy walkthrough that every topic in this
course follows.
