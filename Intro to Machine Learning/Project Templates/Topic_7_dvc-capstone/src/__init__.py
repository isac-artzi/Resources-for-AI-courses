"""FeatureForge -- an introductory feature-engineering + DVC capstone template.

The package keeps the machine-learning logic free of Streamlit so it is easy to
unit-test and to run as reproducible DVC pipeline stages:

* ``paths``     -- every file-system path in one place.
* ``db``        -- open SQLite and read the raw table into pandas.
* ``prepare``   -- DVC stage 1: clean the raw data.
* ``featurize`` -- DVC stage 2: baseline vs engineered feature builders.
* ``train``     -- DVC stage 3: cross-validate baseline vs engineered, fit,
                   and score a single house.
* ``persist``   -- save derived DataFrames as dated CSV files.
* ``ui_data``   -- the ONLY module that imports Streamlit (adds caching).
"""
