"""Source package for the ClassifierLab product.

Modules:
    paths   -- one place that knows where every file lives.
    db      -- open a SQLite connection and read tables into pandas.
    model   -- build, train, and compare kNN / SVM / decision-tree pipelines.
    persist -- save derived DataFrames as versioned CSV files.
"""
