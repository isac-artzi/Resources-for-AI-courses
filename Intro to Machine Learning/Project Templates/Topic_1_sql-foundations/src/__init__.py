"""Source package for the bookshop analytics product.

Modules:
    paths     -- one place that knows where every file lives.
    db        -- open a SQLite connection and read tables/queries into pandas.
    queries   -- the six single-table SELECT queries (basic SQL only).
    analysis  -- all multi-table work, done in pandas with DataFrame.merge.
    clean     -- reusable pandas cleaning helpers.
    persist   -- save derived DataFrames as versioned CSV files.
"""
