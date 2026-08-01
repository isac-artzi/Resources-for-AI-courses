"""
The standing guard. If this fails, the build fails, and that is the point.

`import torch` costs roughly 490 MB of resident memory against Streamlit
Community Cloud's 690 MB guarantee — most of the budget spent before a weight
is loaded, on bundled CUDA libraries a CPU application never calls. The whole
serving stack in this course measures 82 MB without it.

The test asserts on the process, not on the source, because the failure mode
it catches is a TRANSITIVE import: you add a helper to shared/, the helper
imports something from train/, and train/ imports torch. Grepping api/ would
not catch that. Importing the app does.
"""

from __future__ import annotations

import sys


def test_torch_absent_after_importing_the_app():
    import api.main  # noqa: F401

    assert "torch" not in sys.modules, (
        "The deployed application imported PyTorch. Find the import chain with:\n"
        "    python -X importtime -c 'import api.main' 2>&1 | grep -i torch\n"
        "The usual cause is a serving module importing something from train/."
    )


def test_torch_absent_after_importing_the_ui_helpers():
    import ui.service as _svc  # noqa: F401

    assert "torch" not in sys.modules


def test_version_endpoint_reports_it_honestly(client):
    body = client.get("/version").json()
    assert body["torch_imported"] is False
