# Tutorial — Build & Deploy the SQL Foundations Product

This is the **canonical walkthrough** for the course template. Topic 1 uses it
end to end; later topics reuse the same structure and only swap the model and
the analysis. Read it once, carefully, and you will understand every assignment
that follows.

**Audience:** first-semester students. No prior database, pandas, or web-app
experience is assumed. Every step is spelled out.

## Table of contents

1. [What you are building](#1-what-you-are-building)
2. [Prerequisites & accounts](#2-prerequisites--accounts)
3. [The universal build pattern](#3-the-universal-build-pattern)
4. [Project layout](#4-project-layout)
5. [Step-by-step local setup](#5-step-by-step-local-setup)
6. [Understanding each module](#6-understanding-each-module)
7. [Running the tests and linter](#7-running-the-tests-and-linter)
8. [Deploying to Streamlit Community Cloud](#8-deploying-to-streamlit-community-cloud)
9. [The Git & GitHub workflow](#9-the-git--github-workflow)
10. [Reusing this pattern in later topics](#10-reusing-this-pattern-in-later-topics)

---

## 1. What you are building

A **product**, not a homework file. Specifically: a live Streamlit web app that
loads a small bookshop database, answers business questions about it, and
documents itself. The point of Topic 1 is not the bookshop data — it is learning
the *pattern* you will use for every model in this course.

## 2. Prerequisites & accounts

- **Python 3.11+** — https://www.python.org/downloads/
- **A code editor** — VS Code (recommended) or PyCharm Community.
- **A GitHub account** — https://github.com — every assignment is a repo.
- **A Streamlit Community Cloud account** — https://streamlit.io/cloud —
  sign in with GitHub; the free tier is all you need.
- **(Optional) DB Browser for SQLite** — https://sqlitebrowser.org — a GUI to
  peek inside `.sqlite` files while you develop.

## 3. The universal build pattern

Every assignment moves data through the same five stages:

```
  data/raw/*.csv                     (1) raw input, shipped with the repo
        |  db/build_sqlite.py        (2) load into SQLite with basic CRUD
        v
  db/bookshop.sqlite                 a single-file database
        |  pandas.read_sql           (3) read each table into a DataFrame
        v
  DataFrames                         one per table
        |  DataFrame.merge/groupby   (4) join + analyze IN PANDAS (not SQL)
        v
  data/derived/*.csv                 (5) persist derived state (dated CSV)
        |  streamlit
        v
  the web app                        pages re-load the derived CSVs
```

Two rules you must never break in this course:

- **SQL is for basic, single-table CRUD only.** No JOINs / subqueries / window
  functions in SQL.
- **All joining and aggregation happens in pandas.** `merge` is your JOIN.

## 4. Project layout

```
Topic_1_sql-foundations/
├── app.py                  # Streamlit home page (run this)
├── pages/                  # one file per page; Streamlit lists them in the sidebar
│   ├── 1_SQL_Basics.py
│   ├── 2_Top_Customers.py
│   ├── 3_Genre_Popularity.py
│   ├── 4_Revenue_By_Employee.py
│   └── 5_Data_Profile.py
├── src/                    # the reusable, tested Python package
│   ├── paths.py            # where everything lives
│   ├── db.py               # open SQLite + read into pandas
│   ├── queries.py          # the six single-table SELECTs
│   ├── analysis.py         # all pandas merge/groupby analysis
│   ├── clean.py            # reusable cleaning helpers
│   ├── persist.py          # save derived DataFrames as dated CSV
│   └── ui_data.py          # Streamlit-cached data loaders
├── db/
│   ├── build_sqlite.py     # CREATE TABLE + INSERT (CRUD)
│   └── crud_maintenance.py # UPDATE + DELETE (CRUD)
├── data/
│   ├── raw/                # input CSVs (+ provenance README)
│   └── derived/            # generated CSVs (git-ignored)
├── tests/                  # pytest suite (mirrors src/ and db/)
├── .github/workflows/ci.yml# runs ruff + pytest on every push
├── .streamlit/config.toml  # app theme/config for Streamlit Cloud
├── requirements.txt        # runtime deps (what Streamlit Cloud installs)
├── requirements-dev.txt    # + test/lint deps
├── ruff.toml               # linter config
└── pytest.ini              # test config
```

## 5. Step-by-step local setup

```bash
# From inside Topic_1_sql-foundations/

# 1. Create an isolated environment so this project's packages don't collide
#    with other projects on your machine.
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install everything (runtime + dev tools).
pip install -r requirements-dev.txt

# 3. Build the database. Watch it print one line per table.
python db/build_sqlite.py

# 4. (Optional) See UPDATE and DELETE in action.
python db/crud_maintenance.py

# 5. Launch the app. Your browser opens at http://localhost:8501
streamlit run app.py
```

Click through the sidebar pages. Everything you see is produced by the modules
in `src/`.

## 6. Understanding each module

- **`src/paths.py`** — computes every path relative to the repo root, so the
  project runs from any folder.
- **`db/build_sqlite.py`** — reads each raw CSV and writes it into SQLite using
  `CREATE TABLE` + `INSERT`. It is *idempotent*: re-running drops and rebuilds,
  so you always get the same clean database.
- **`src/db.py`** — `get_connection()` opens the file; `read_query()` /
  `read_table()` bring rows into pandas. Note the use of `?` placeholders — the
  safe way to pass values into SQL.
- **`src/queries.py`** — the six basic single-table queries, stored as text so
  the UI can display them next to their results.
- **`src/analysis.py`** — the interesting part: `merge` joins tables and
  `groupby` aggregates, all in pandas. These functions take and return plain
  DataFrames, which makes them trivial to unit-test.
- **`src/clean.py`** — small, single-purpose cleaning helpers, each returning a
  new DataFrame (never mutating its input).
- **`src/persist.py`** — writes derived DataFrames to `data/derived/` with a
  dated filename like `cleaned_top_customers_20240115.csv`.
- **`src/ui_data.py`** — the only module that imports Streamlit; it adds caching
  so the database builds once and tables load once per session.

## 7. Running the tests and linter

CI runs these on every push; run them locally first so pushes stay green.

```bash
pytest -q          # 22 tests should pass
ruff check .       # style + correctness; "All checks passed!"
ruff check --fix . # auto-fix simple issues (import order, whitespace)
```

If a test fails, read the message: it names the file, the line, and what it
expected. Fix the code, not the test (unless the test is genuinely wrong).

## 8. Deploying to Streamlit Community Cloud

1. Push your repository to GitHub (see the next section).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. **New app** → pick your repo, branch `main`, and main file `app.py`.
4. Click **Deploy**. Streamlit installs `requirements.txt` and runs the app.
5. When it goes live, copy the public URL and paste it at the top of your
   `README.md`.

> The app builds the SQLite database automatically on first load (`ui_data.py`
> calls the build step if the file is missing), so you do **not** commit the
> `.sqlite` file — it is generated on the server.

## 9. The Git & GitHub workflow

```bash
git init
git add .
git commit -m "Topic 1: initial SQL foundations product"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Work in small, well-described commits. For team assignments, each member works
on a **feature branch**, opens a **pull request**, and merges after review. Tag
your final submission: `git tag v1.0.0 && git push --tags`.

## 10. Reusing this pattern in later topics

Every later assignment keeps this exact skeleton and changes only the middle:

| Layer | Stays the same | Changes per topic |
|-------|----------------|-------------------|
| `db/build_sqlite.py` | CRUD load into SQLite | the raw dataset |
| `src/db.py`, `paths.py`, `persist.py` | unchanged | — |
| `src/analysis.py` | pandas merge/groupby | the actual model (scikit-learn) |
| `pages/` | one page per question | topic-specific views |
| `tests/` | same structure | assertions about the new model |

So once you understand Topic 1, you understand the plumbing for Topics 2–7.
The only new thing each week is the machine-learning idea itself.
