"""LatentLens -- an introductory unsupervised-learning product template.

The package is organised so the machine-learning logic never depends on
Streamlit, which keeps it easy to unit-test:

* ``paths``   -- every file-system path in one place.
* ``db``      -- open SQLite and read tables into pandas.
* ``cluster`` -- scale features and group shoppers with k-means.
* ``reduce``  -- squeeze many features into two with PCA (for plotting).
* ``rules``   -- mine "buy X -> buy Y" association rules from baskets.
* ``persist`` -- save derived DataFrames as dated CSV files.
* ``ui_data`` -- the ONLY module that imports Streamlit (adds caching).
"""
