"""Source package for the Honest Forecast product.

Modules:
    paths    -- one place that knows where every file lives.
    db       -- open a SQLite connection and read tables into pandas.
    features -- build day-index features and split the series by time.
    model    -- train, evaluate, and forecast with linear regression.
    persist  -- save derived DataFrames as versioned CSV files.
"""
