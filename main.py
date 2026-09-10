"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Entry Point
Description : Interactive entry point for forward and inverse ERP neural operators
==================================================

Run:
    python main.py

All menus, prompts, and orchestration live in utils/cli.py. Dataset
preprocessing is handled by utils/erp_dataset.py, forward operator
training/evaluation by forward_operators/neural_operator_utils.py, inverse-model
training/evaluation by inverse_operators/, and all figure creation/saving
by utils/plotting.py.
"""

from __future__ import annotations

from utils.cli import main

if __name__ == "__main__":
    results = main()
