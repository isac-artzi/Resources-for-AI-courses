# Topic 2 — Data Quality: Data-Quality Dashboard

> **Live app:** _paste your Streamlit Community Cloud URL here after deploying_
> **Repository:** _paste your GitHub repository URL here_

A starter **product** template for an introductory machine-learning course.
It is a small, deployed Streamlit application that profiles a deliberately
messy dataset, validates it against a written schema, and cleans it with a
re-runnable pipeline. It reuses the **universal build pattern** from Topic 1:

```
raw CSV  ->  SQLite (basic CRUD)  ->  pandas (profile/clean)  ->  derived CSV  ->  Streamlit UI
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

You are handed a `signups` export with all the usual real-world defects:
missing ages and incomes, impossible values (an age of 999, a negative
income), inconsistent country and plan spellings (`usa `, `Canada`, `PRO`),
and exact duplicate rows. Before anyone builds a model on this data, someone
has to answer: *how bad is it, and can we make it trustworthy?* The goal is a
dashboard that **measures**, **validates**, and **cleans** the data.

## Theory

**Profiling** is measuring a dataset before you trust it: row and column
counts, data types, missing-value counts, duplicate counts, and numeric ranges.
See `src/profile.py`.

**A schema is a written contract** for what clean data looks like — each
column's type and the values it may hold. The `pandera` library turns that
contract into runnable checks. The key idea of this topic: the **raw** data
*fails* the schema, and the **cleaned** data *passes* it. See `src/schema.py`.

**Cleaning is a pipeline of small, single-purpose steps**, each returning a new
DataFrame (never mutating its input) so it is easy to test and reason about:
drop duplicates → fix types → standardize text → mark impossible values missing
→ impute the gaps with the median. See `src/clean.py`.

**Why persist to CSV?** Streamlit Community Cloud's file system is *ephemeral*,
so we write cleaned results to a dated CSV in `data/derived/` (the course
convention for derived state).

## Data

A **synthetic** `signups` dataset lives in `data/raw/signups.csv`, generated
deterministically (fixed seed) by `db/generate_raw.py` so the tests are
reproducible. The messiness — missing values, outliers, inconsistent text, and
15 appended duplicate rows — is injected on purpose. A tiny `provenance` table
records where the data came from.

## Method

The data flows through the build pattern:

1. **Load** — `db/build_sqlite.py` reads the CSV into SQLite (`CREATE TABLE` +
   `INSERT`) and writes a `provenance` table.
2. **Read** — `src/db.py` reads the table into a pandas DataFrame.
3. **Profile** — `src/profile.py` measures shape, types, missing values, and
   ranges (the **Raw Profile** page).
4. **Validate** — `src/schema.py` checks the data against the pandera schema
   (the **Schema Report** page; raw data fails).
5. **Clean** — `src/clean.py` runs the pipeline; the cleaned data passes the
   same schema (the **Cleaned Explorer** page).
6. **Persist** — `src/persist.py` writes the cleaned DataFrame to a dated CSV.

## Results

The deployed app has three pages: **Raw Profile** (the mess, measured),
**Schema Report** (the raw data failing validation, rule by rule), and
**Cleaned Explorer** (the pipeline output passing the schema, with a
before/after summary and a save button).

## Ethics

Cleaning choices are ethical choices. Imputing missing incomes with the median
is convenient, but it hides uncertainty and can mask systematic gaps (e.g. if
income is missing more often for one country). Dropping "impossible" rows can
silently discard real edge cases. Document every rule you apply and state its
limits, so downstream users know what the cleaned data does and does not
represent. _(Replace this section with your own ~300-word reflection when you
adapt the template.)_

## References

- pandera — data validation: https://pandera.readthedocs.io/
- pandas — working with missing data: https://pandas.pydata.org/docs/user_guide/missing_data.html
- pandas — `read_sql`: https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html
- Streamlit — multipage apps: https://docs.streamlit.io/develop/concepts/multipage-apps

---

See [`../Topic_1_sql-foundations/TUTORIAL.md`](../Topic_1_sql-foundations/TUTORIAL.md)
for the full, step-by-step build-and-deploy walkthrough that every topic in this
course follows.
