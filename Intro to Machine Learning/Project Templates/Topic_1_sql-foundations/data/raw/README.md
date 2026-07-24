# Raw data

This folder holds the **raw input** for the project: a small, synthetic
online-bookshop dataset. Using a tiny built-in dataset keeps the template fast
and lets the tests run offline in seconds. When you start your own assignment,
replace these CSVs with a real dataset and record its provenance below.

## Provenance

| Field | Value |
|-------|-------|
| Source | Synthetic sample bundled with this template |
| License | Public domain (synthetic; safe to redistribute) |
| Retrieved | Generated for this template |

## Tables

| File | Rows | Description |
|------|------|-------------|
| `genres.csv` | 5 | Book genres (Fiction, Mystery, ...) |
| `authors.csv` | 4 | Authors |
| `series.csv` | 6 | Book series, each linked to an author |
| `books.csv` | 15 | Individual books, linked to a series and a genre |
| `employees.csv` | 3 | Sales representatives |
| `customers.csv` | 8 | Customers, each assigned a support rep |
| `invoices.csv` | 12 | One row per purchase |
| `invoice_items.csv` | 24 | Line items: which book was bought on which invoice |

These tables are **linked by id columns** (for example `books.genre_id`
points at `genres.genre_id`). In this course we do NOT join them in SQL --
we read each table into its own pandas DataFrame and join with
`DataFrame.merge`. See `src/analysis.py`.
