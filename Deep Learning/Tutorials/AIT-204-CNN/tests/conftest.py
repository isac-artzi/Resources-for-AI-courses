"""Run the tests against the student package (default) or the instructor solutions.

    pytest                      # your code
    LAB_TARGET=solutions pytest # reference implementation
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
