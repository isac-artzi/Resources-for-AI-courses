# Topic 7 — Feature Engineering & Reproducible Pipelines: FeatureForge

> **Live app:** _paste your Streamlit Community Cloud URL here after deploying_
> **Repository:** _paste your GitHub repository URL here_
> **Release:** v1.0.0

A capstone **product** template for an introductory machine-learning course. It
takes a house-price model and makes it **measurably better and fully
reproducible**: it engineers features, proves they help with a fair comparison,
and wires the whole thing together as a **DVC** pipeline. It reuses the
**universal build pattern** from Topic 1:

```
raw CSV  ->  SQLite (basic CRUD)  ->  pandas + sklearn  ->  Streamlit UI
                   \__ prepare -> featurize -> train (DVC pipeline) __/
```

This README doubles as **release documentation**: the standard report sections
(Problem, Theory, Data, Method, Results, Ethics, References) followed by
**Release Notes** and a **Changelog**.

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

### Reproduce the pipeline with DVC (optional)

```bash
pip install dvc          # not needed to run the app or tests
dvc init                 # once, in your own repo
dvc repro                # prepare -> featurize -> train, only re-running what changed
dvc metrics show         # print the baseline vs engineered scores
```

> The app also computes the comparison live, so you can explore it without DVC.
> `dvc repro` produces the *same* numbers as version-controlled, reproducible
> files (`data/derived/metrics.json`).

---

## Problem

A first model is rarely the final one. Given a working house-price predictor,
two questions decide whether it is production-ready: **can we make it more
accurate**, and **can anyone reproduce the result** from the raw data months
later? This capstone answers both.

## Theory

**Feature engineering** reshapes raw columns into inputs a model can use well:

- **One-hot encoding** gives each category its own on/off column, instead of a
  fake number line (Ashby = 0, Brook = 1, …) that invents an order.
- **Log transforms** (`log1p`) pull in the long right tail of skewed columns
  like living area and lot size.
- **Binning** turns a messy continuous effect (house age) into a few clear
  eras.
- **Ratio features** combine columns into something more informative than
  either alone (square feet *per bedroom*).

**Fair comparison** means changing only the features: the same linear model and
the same 5-fold cross-validation score both the baseline and the engineered
inputs, so the R² gain is attributable to engineering, not luck.

**Reproducible pipelines.** **DVC** records a hash of every stage's inputs and
outputs. `dvc repro` re-runs a stage only when something it depends on changed,
so results rebuild exactly and cheaply — the data-science version of a build
system.

## Data

A **synthetic** 1500-row house dataset (`data/raw/houses.csv`) with a
categorical `neighborhood`, skewed size columns, and a `price` target, generated
deterministically by `db/generate_raw.py`. The neighborhood premiums are
deliberately out of alphabetical order so naive label-encoding fails. See
[`data/raw/README.md`](data/raw/README.md).

## Method

1. **Load** — `db/build_sqlite.py` reads the CSV into SQLite plus a `provenance`
   table.
2. **Prepare** (DVC stage 1) — `src/prepare.py` cleans the data → `prepared.csv`.
3. **Featurize** (DVC stage 2) — `src/featurize.py` builds engineered features
   → `features.csv`.
4. **Train** (DVC stage 3) — `src/train.py` cross-validates baseline vs
   engineered, saves `metrics.json` and the fitted `models/model.joblib`.
5. **Serve** — the Streamlit app shows the comparison and scores single houses.

## Results

Feature engineering lifts cross-validated **R² from ~0.48 (baseline) to ~0.93
(engineered)** on the synthetic data — a large, defensible improvement driven
mostly by one-hot encoding the neighborhood. The app's four pages present the
data, the baseline vs engineered feature tables, the head-to-head scores, and a
single-house price estimator.

## Ethics

A price model trained on neighborhood can quietly encode the biases baked into
housing history — a "neighborhood premium" may be a proxy for demographics, not
just amenities. Reproducibility is an ethical tool as well as a technical one:
`dvc repro` lets an auditor rebuild and question exactly how a number was
produced. Document what each feature stands for, and be cautious using such a
model to *set* prices rather than merely estimate them. _(Replace this section
with your own ~300-word reflection when you adapt the template.)_

## References

- scikit-learn — preprocessing (encoding, scaling): https://scikit-learn.org/stable/modules/preprocessing.html
- scikit-learn — cross-validation: https://scikit-learn.org/stable/modules/cross_validation.html
- pandas — `get_dummies`: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
- DVC — get started / pipelines: https://dvc.org/doc/start/data-pipelines

---

## Release Notes — v1.0.0

First stable release of FeatureForge.

- **Engineered model beats baseline** by a wide, cross-validated margin
  (R² ≈ 0.48 → 0.93).
- **Reproducible three-stage DVC pipeline** (prepare → featurize → train) with
  tracked metrics.
- **Four-page Streamlit app**: Dataset, Feature Engineering, Pipeline Results,
  Predict Price.
- **Green CI**: `pytest` + `ruff` on every push.

## Changelog

### v1.0.0
- Added the DVC pipeline (`dvc.yaml`) with `prepare`, `featurize`, and `train`
  stages and tracked `metrics.json`.
- Added four feature-engineering techniques: one-hot encoding, log transforms,
  age binning, and ratio features.
- Added the baseline-vs-engineered cross-validation comparison.
- Added the single-house price estimator using the same feature builder as
  training.

### v0.1.0 (baseline)
- Initial house-price model on raw numeric features only.

---

See [`../Topic_1_sql-foundations/TUTORIAL.md`](../Topic_1_sql-foundations/TUTORIAL.md)
for the full, step-by-step build-and-deploy walkthrough that every topic in this
course follows.
