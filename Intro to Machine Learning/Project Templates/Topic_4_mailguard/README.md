# Topic 4 — Bayesian ML: Mailguard

> **Live app:** _paste your Streamlit Community Cloud URL here after deploying_
> **Repository:** _paste your GitHub repository URL here_

A starter **product** template for an introductory machine-learning course.
It is a small, deployed Streamlit application that classifies email as **spam**
or **ham** with **naive Bayes**, and explains each decision. It reuses the
**universal build pattern** from Topic 1:

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

Inboxes are flooded with spam. We want a filter that flags spam automatically
*and* can explain itself — a black box that silently deletes mail is not
trustworthy. The goal is a live app that scores any message and shows why.

## Theory

**Naive Bayes** applies Bayes' theorem with a simplifying assumption: that each
word is independent evidence of spam. It multiplies the per-word evidence
together to estimate `P(spam | words)`. The assumption is technically false for
language, yet the model is fast, needs little data, and works well — a useful
lesson in pragmatic modelling.

**Text to numbers.** `TfidfVectorizer` converts each message into a vector of
word weights (frequent-everywhere words count for less). We chain it with
`MultinomialNB` in a single scikit-learn `Pipeline` so the exact same
preprocessing runs at training and prediction time.

**Laplace smoothing** (`alpha=1.0`) adds one to every word count so a single
never-before-seen word cannot force a probability of exactly zero.

**Explainability.** Naive Bayes stores each word's log-probability per class;
the spam-minus-ham difference tells us how much a word tips the scale, which we
surface as the "most spam-indicative words".

## Data

A **synthetic**, balanced 800-message corpus (`data/raw/emails.csv`) generated
deterministically by `db/generate_raw.py`. Messages are drawn from spammy,
hammy, and neutral word lists with mild cross-class leakage so the task is
realistic. See [`data/raw/README.md`](data/raw/README.md).

## Method

1. **Load** — `db/build_sqlite.py` reads the CSV into SQLite (`CREATE` +
   `INSERT`) plus a `provenance` table.
2. **Read** — `src/db.py` reads the table into a pandas DataFrame.
3. **Split** — `src/model.py` makes a stratified, seeded train/test split.
4. **Train** — the tf-idf + naive-Bayes `Pipeline` is fitted on the training
   messages.
5. **Evaluate** — accuracy, precision, recall, F1, ROC-AUC, and a confusion
   matrix on the held-out test messages.
6. **Explain** — `influential_words` surfaces the strongest spam signals; the
   app scores any pasted message.

## Results

The deployed app has three pages: **Dataset** (browse the corpus and class
balance), **Train & Evaluate** (the metrics and confusion matrix), and
**Classify a Message** (paste text for a spam probability plus the words that
drove the score).

## Ethics

A spam filter makes mistakes with unequal costs. A **false positive** — real
mail wrongly flagged as spam — can hide a job offer or a medical result, which
is usually worse than a **false negative** (spam that slips through). Report
precision and recall separately, not just accuracy, and be transparent about
which errors the model makes. The explanation view exists so decisions can be
questioned rather than trusted blindly. _(Replace this section with your own
~300-word reflection when you adapt the template.)_

## References

- scikit-learn — naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
- scikit-learn — `TfidfVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- scikit-learn — classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- Streamlit — multipage apps: https://docs.streamlit.io/develop/concepts/multipage-apps

---

See [`../Topic_1_sql-foundations/TUTORIAL.md`](../Topic_1_sql-foundations/TUTORIAL.md)
for the full, step-by-step build-and-deploy walkthrough that every topic in this
course follows.
