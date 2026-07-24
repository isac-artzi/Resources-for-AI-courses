# Topic 3 — Linear Regression: Honest Forecast

> **Live app:** _paste your Streamlit Community Cloud URL here after deploying_
> **Repository:** _paste your GitHub repository URL here_

A starter **product** template for an introductory machine-learning course.
It is a small, deployed Streamlit application that fits a **linear regression**
to a daily price series and projects it forward — *honestly*. It reuses the
**universal build pattern** from Topic 1:

```
raw CSV  ->  SQLite (basic CRUD)  ->  pandas + sklearn  ->  derived CSV  ->  Streamlit UI
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

Given a daily price series, produce a forward-looking forecast a stakeholder
can *trust*. The trap most beginners fall into is scoring a time-series model
on randomly shuffled data — which secretly lets the model peek at the future
and reports a dishonestly high accuracy. The goal here is a forecast that is
evaluated the way it will actually be used, and that admits its uncertainty.

## Theory

**Linear regression** fits a straight line `close ≈ slope · t + intercept` by
ordinary least squares. Here `t` is a simple day index (0, 1, 2, …). We use
scikit-learn's `LinearRegression`.

**Chronological splitting** is the heart of an honest forecast. For ordinary
data you shuffle before splitting; for a time series that leaks future
information into training. Instead we train on the earliest days and test on the
most recent days — exactly how a real forecast is judged. See
`chronological_split` in `src/features.py`.

**Honest metrics.** *R²* is the fraction of variation explained (1.0 perfect,
0.0 no better than the mean); *RMSE* is the typical error in dollars.

**Uncertainty band.** A single predicted line pretends we know the future
exactly. We surround the forecast with ± 2 standard deviations of the model's
past errors (a rough 95% interval), so the band — not the center line — is the
real message.

## Data

A **synthetic** 500-day price series lives in `data/raw/prices.csv`, generated
deterministically by `db/generate_raw.py` as trend + mild seasonality + noise.
Using synthetic data keeps the tests fast and offline; the file documents how to
swap in real `yfinance` data. See [`data/raw/README.md`](data/raw/README.md).

## Method

1. **Load** — `db/build_sqlite.py` reads the CSV into SQLite (`CREATE` +
   `INSERT`) plus a `provenance` table.
2. **Read** — `src/db.py` reads the table into a pandas DataFrame.
3. **Feature** — `src/features.py` parses dates, adds the day index `t`, and
   splits the series by **time**.
4. **Train** — `src/model.py` fits `LinearRegression` on the training days.
5. **Evaluate** — `src/model.py` scores it (R², RMSE) on the held-out future.
6. **Forecast** — `src/model.py` projects the trend forward with a ± 2σ band.
7. **Persist** — `src/persist.py` writes the forecast to a dated CSV.

## Results

The deployed app has three pages: **Price History** (the raw series and its
trend), **Train & Evaluate** (time-split fit with R² and RMSE, and a fitted-vs-
actual chart), and **Forecast** (a forward projection with an uncertainty band
and a save button).

## Ethics

Forecasts influence decisions, so overstating confidence is a real harm. This
template deliberately (1) evaluates on the future, not shuffled data, and
(2) always shows an uncertainty band. When adapting it, resist the temptation
to report the single best-looking number: state what the model assumes (that the
past trend simply continues), what it cannot see (news, shocks), and how wide
its errors really are. _(Replace this section with your own ~300-word reflection
when you adapt the template.)_

## References

- scikit-learn — `LinearRegression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- scikit-learn — model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- pandas — `read_sql`: https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html
- Streamlit — multipage apps: https://docs.streamlit.io/develop/concepts/multipage-apps

---

See [`../Topic_1_sql-foundations/TUTORIAL.md`](../Topic_1_sql-foundations/TUTORIAL.md)
for the full, step-by-step build-and-deploy walkthrough that every topic in this
course follows.
