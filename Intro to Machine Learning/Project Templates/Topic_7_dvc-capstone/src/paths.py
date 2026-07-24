"""Central list of file-system paths used across the project.

Keeping every path in ONE module means the rest of the code never has to
guess where things live, and if you move a folder you only edit it here.

We compute paths *relative to this file* using ``pathlib``. That way the
project works no matter what directory you launch it from -- on your laptop
or on Streamlit Community Cloud.
"""

from pathlib import Path

# ``__file__`` is this file (src/paths.py). ``.resolve()`` makes it an
# absolute path; ``.parent.parent`` walks up from ``src/`` to the repo root.
ROOT = Path(__file__).resolve().parent.parent

# Raw input CSVs that ship with the template (never modified at runtime).
RAW_DIR = ROOT / "data" / "raw"

# Derived/cleaned CSVs that the app writes and re-reads.
DERIVED_DIR = ROOT / "data" / "derived"

# The SQLite database that db/build_sqlite.py creates from the raw CSVs.
DB_PATH = ROOT / "db" / "homes.sqlite"

# --- DVC pipeline artifacts -------------------------------------------------
# The reproducible pipeline (dvc.yaml) reads the raw CSV and writes these files,
# one per stage. Keeping their locations here lets both the stage scripts and
# the Streamlit app agree on where each artifact lives.
RAW_CSV = RAW_DIR / "houses.csv"
PREPARED_CSV = DERIVED_DIR / "prepared.csv"  # stage 1: prepare
FEATURES_CSV = DERIVED_DIR / "features.csv"  # stage 2: featurize (engineered)
METRICS_JSON = DERIVED_DIR / "metrics.json"  # stage 3: train (comparison)
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"  # stage 3: the fitted engineered model
