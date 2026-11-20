"""Shared setup of the Module_PlaneFR tests: makes planefr_lib importable
(it lives in notebooks/, not installed as a package).

Run from the root of Module_PlaneFR:

    py -m pytest tests
"""

import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))
