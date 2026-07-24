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
DB_PATH = ROOT / "db" / "quality.sqlite"
