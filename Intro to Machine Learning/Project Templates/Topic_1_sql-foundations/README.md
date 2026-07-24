# Topic 1 — SQL Foundations: Bookshop Analytics

> **Live app:** _paste your Streamlit Community Cloud URL here after deploying_
> **Repository:** _paste your GitHub repository URL here_

A starter **product** template for an introductory machine-learning course.
It is a small, deployed Streamlit application that answers business questions
about an online bookshop — and, more importantly, it demonstrates the
**universal build pattern** every assignment in this course reuses:

```
raw CSVs  ->  SQLite (basic CRUD)  ->  pandas (merge/clean)  ->  derived CSV  ->  Streamlit UI
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

# 3. Build the SQLite database from the raw CSVs (basic CRUD).
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

A fictional bookshop shares its operational database and asks for an
internal analytics tool the sales, marketing, and finance teams can open in a
browser to answer recurring questions:

- Who are the top-spending customers?
- Which genres bring in the most revenue?
- Which employee (sales rep) is tied to which revenue line?

The goal is **not a query log** — it is a live application a
non-technical stakeholder can use and trust.

## Theory

**Basic CRUD** is the four fundamental database operations: **C**reate a table,
**R**ead rows (SELECT), **U**pdate rows, and **D**elete rows. In this course we
use CRUD on **single tables only** — no JOINs, subqueries, or window functions
in SQL. See `db/build_sqlite.py` (CREATE + INSERT) and `db/crud_maintenance.py`
(UPDATE + DELETE).

**`DataFrame.merge` is the pandas equivalent of a SQL JOIN.** It lines up rows
from two tables using a shared key column. We do all joining in pandas because,
at this introductory level, it is easier to read, test, and debug than
multi-table SQL — and it keeps SQL focused on the basics. Example:

```python
# "For each line item, attach the customer it belongs to" (a JOIN in pandas):
items = invoice_items.merge(invoices, on="invoice_id").merge(customers, on="customer_id")
```

**Why persist to CSV?** Streamlit Community Cloud's file system is *ephemeral* —
anything the app writes is lost on restart, and writes to the bundled SQLite
file do not survive a redeploy. So the convention is: read from SQLite, clean
and join in pandas, and write derived state to a dated CSV in `data/derived/`.

## Data

A small **synthetic** bookshop dataset lives in `data/raw/`. Using tiny
built-in data keeps the app fast and lets the tests run offline in seconds.
Eight linked tables: genres, authors, series, books, employees, customers,
invoices, invoice_items. Full provenance and a table dictionary are in
[`data/raw/README.md`](data/raw/README.md).

## Method

The data flows through the build pattern:

1. **Load** — `db/build_sqlite.py` reads each CSV and writes it to SQLite with
   `CREATE TABLE` + `INSERT` (one table at a time).
2. **Read** — `src/db.py` reads each table into its own pandas DataFrame via
   `pandas.read_sql`.
3. **Query** — `src/queries.py` holds the six basic single-table SELECT queries
   (shown on the **SQL Basics** page).
4. **Analyze** — `src/analysis.py` does all joining and aggregation in pandas
   (`merge`, `groupby`).
5. **Clean** — `src/clean.py` collects reusable cleaning helpers.
6. **Persist** — `src/persist.py` writes derived DataFrames to dated CSVs.
7. **Present** — `app.py` + `pages/` render the results in Streamlit.

## Results

The deployed app has five pages: **SQL Basics** (the six queries), **Top
Customers**, **Genre Popularity**, **Revenue by Employee** (each answering one
business question with a table, a chart, and a plain-English takeaway), and a
**Data Profile** page (shape, types, and null checks for every table).

## Ethics

Every analytical choice — which rows to include, which to drop, which table to
join with which — shapes the story the numbers tell. For example, the
"Revenue by Employee" page reflects the revenue of each rep's **assigned**
customers, not individual sales skill; reporting it as a performance ranking
would be misleading. Represent the underlying population honestly and state the
limits of each claim. _(Replace this section with your own ~300-word reflection
when you adapt the template.)_

## References

- pandas — `read_sql`: https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html
- pandas — `merge`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html
- SQLite — SQL syntax: https://www.sqlite.org/lang.html
- Streamlit — multipage apps: https://docs.streamlit.io/develop/concepts/multipage-apps

---

See [`TUTORIAL.md`](TUTORIAL.md) for a full, step-by-step build-and-deploy walkthrough.
