"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Entry Point
Description : Interactive entry point for the ERP, Displacement-field, and
              invertible (iFNO/iDCO/iGNO) neural-operator problems
==================================================

Run:
    python main.py

Prompts for which problem to work on first -- ERP (config <-> ERP
spectrum, forward/inverse), Displacement field (config,freq,position ->
velocity field), or an invertible operator (joint forward+inverse on the
ERP problem, one of iFNO/iDCO/iGNO) -- then dispatches accordingly. All
menus, prompts, and orchestration live in utils/cli.py. Dataset
preprocessing is handled by utils/erp_dataset.py (ERP) and
utils/field_dataset.py (displacement); training/evaluation lives in
erp_forward/scripts/, erp_inverse/scripts/,
disp_forward/scripts/, and erp_invertible/scripts/
respectively; all figure creation/saving goes through utils/plotting.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Dataset/checkpoint paths are stored relative to the repository root, so
# always run from there -- `python path/to/main.py` works from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import main  # noqa: E402

if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
