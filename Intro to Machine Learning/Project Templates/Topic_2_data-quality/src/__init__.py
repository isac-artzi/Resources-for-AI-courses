"""Source package for the Data-Quality Dashboard product.

Modules:
    paths    -- one place that knows where every file lives.
    db       -- open a SQLite connection and read tables into pandas.
    profile  -- measure data quality (shape, types, missing values, ranges).
    clean    -- a modular, re-runnable cleaning pipeline.
    schema   -- a pandera contract for what clean data looks like.
    persist  -- save derived DataFrames as versioned CSV files.
"""
