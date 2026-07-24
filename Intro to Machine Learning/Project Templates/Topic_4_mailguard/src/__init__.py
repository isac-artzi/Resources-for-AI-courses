"""Source package for the Mailguard product.

Modules:
    paths   -- one place that knows where every file lives.
    db      -- open a SQLite connection and read tables into pandas.
    model   -- the tf-idf + naive-Bayes pipeline, metrics, and explanations.
    persist -- save derived DataFrames as versioned CSV files.
"""
