# Topic 6 — Clustering & Dimensionality Reduction: LatentLens

> **Live app:** _paste your Streamlit Community Cloud URL here after deploying_
> **Repository:** _paste your GitHub repository URL here_

A starter **product** template for an introductory machine-learning course.
It is a small, deployed Streamlit application that finds **hidden structure** in
customer data without any labels — grouping shoppers with **k-means**,
mapping them in 2-D with **PCA**, and mining **association rules** from their
shopping baskets. It reuses the **universal build pattern** from Topic 1:

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

Most machine learning you meet first is *supervised* — you have labels and
predict them. But real data often arrives with **no labels**: you just have
customers and want to understand them. Which shoppers are similar? Can we see
the whole customer base on one chart? Which products sell together? These are
**unsupervised** questions, and there is no accuracy score to lean on.

## Theory

**k-means clustering** splits rows into `k` groups so members of a group are as
similar as possible (smallest total distance to their group's centre). You must
choose `k`; the **elbow** (where extra clusters stop reducing the within-cluster
spread much) and the **silhouette** score (separation, from −1 to 1) guide that
choice.

**Principal Component Analysis (PCA)** builds new axes — combinations of the
original features ordered by how much variation they capture — so keeping the
first two lets us plot high-dimensional data on a flat chart.

**Why scale first?** Both k-means and PCA are driven by variance and distance,
so a large-scale feature (income, in the tens of thousands) would swamp a
small-scale one (visits per month). `StandardScaler` puts every feature on the
same footing so each counts equally.

**Association rules** (via the **Apriori** algorithm) find products bought
together. Each `if → then` rule is scored by **support** (how common),
**confidence** (how reliable), and **lift** (how much stronger than chance —
lift > 1 is a genuine association).

## Data

A **synthetic** 900-row shopper dataset (`data/raw/shoppers.csv`) built from
three hidden segments (budget, family, premium). Each shopper has five numeric
features and six `bought_*` product-category flags, generated deterministically
by `db/generate_raw.py`. See [`data/raw/README.md`](data/raw/README.md).

## Method

1. **Load** — `db/build_sqlite.py` reads the CSV into SQLite (`CREATE` +
   `INSERT`) plus a `provenance` table.
2. **Read** — `src/db.py` reads the table into a pandas DataFrame.
3. **Scale** — `src/cluster.py` standardizes the five numeric features.
4. **Cluster** — k-means groups the shoppers; `cluster_scores` sweeps `k` for
   the elbow and silhouette.
5. **Reduce** — `src/reduce.py` runs PCA to a 2-D map for plotting.
6. **Mine** — `src/rules.py` runs Apriori on the basket flags to find rules.

## Results

The deployed app has four pages: **Dataset** (the raw shoppers, features, and
purchase rates), **Clusters** (elbow, silhouette, an interactive `k`, and a
PCA scatter coloured by cluster), **PCA** (explained variance and a 2-D map),
and **Association Rules** (adjustable support/confidence thresholds and the
resulting rules ranked by lift).

## Ethics

Unsupervised segmentation quietly sorts people into groups that can then be
treated differently — who sees which price, which offer, which nothing at all.
A cluster labelled "low value" can become a self-fulfilling prophecy. Segments
built from behaviour can also stand in for protected attributes (age, income)
even when those are excluded. Treat clusters and rules as *hypotheses to
investigate*, not verdicts, and check who is helped or harmed before acting on
them. _(Replace this section with your own ~300-word reflection when you adapt
the template.)_

## References

- scikit-learn — clustering (k-means): https://scikit-learn.org/stable/modules/clustering.html
- scikit-learn — PCA: https://scikit-learn.org/stable/modules/decomposition.html
- scikit-learn — `StandardScaler`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- mlxtend — association rules (Apriori): https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

---

See [`../Topic_1_sql-foundations/TUTORIAL.md`](../Topic_1_sql-foundations/TUTORIAL.md)
for the full, step-by-step build-and-deploy walkthrough that every topic in this
course follows.
