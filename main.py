"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Entry Point
Description : Interactive entry point for the ERP, Displacement-field, and
              iFNO neural-operator problems
==================================================

Run:
    python main.py

Prompts for which problem to work on first -- ERP (config <-> ERP
spectrum, forward/inverse), Displacement field (config,freq,position ->
velocity field), or iFNO (joint forward+inverse on the ERP problem) --
then dispatches accordingly. All menus, prompts, and orchestration live in
utils/cli.py. Dataset preprocessing is handled by utils/erp_dataset.py
(ERP) and utils/field_dataset.py (displacement); training/evaluation lives
in erp_forward_operators/, erp_inverse_operators/,
displacement_forward_operators/, and ifno/ respectively; all figure
creation/saving goes through utils/plotting.py.
"""

from __future__ import annotations

from utils.cli import main

if __name__ == "__main__":
    results = main()
