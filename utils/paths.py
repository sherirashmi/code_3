"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Paths
Description : One place that decides where every checkpoint and plot goes
==================================================

Every ERP workflow is organised by *experiment type* first, then by the
*dataset* it was trained on (``utils.erp_dataset.DATASETS`` tag, e.g.
``10k``, ``100k``, ``200k_18modes`` -- the tag also fixes the plate-mode
basis), then by *model*::

    erp_forward_operators/
        models/<EXPERIMENT>/<dataset>/<model>.pth      EXPERIMENT = GENERAL | FREQ_HOLDOUT
        plots/<EXPERIMENT>/<dataset>/<MODEL>/          per-model figures
        plots/<EXPERIMENT>/<dataset>/ALL_MODELS/       cross-model comparison
        models|plots/DCO_VARIANTS, GNO_VARIANTS, LEGACY  archived experiments
    erp_inverse_operators/
        models/<dataset>/inverse_<model>.pth
        plots/<dataset>/<MODEL>/ and plots/<dataset>/ALL_MODELS/
    erp_invertible_operators/
        models/<dataset>/<model>.pth
        plots/<dataset>/<MODEL>/ and plots/<dataset>/ALL_MODELS/

All paths are absolute (anchored at the repository root), so running
``main.py`` from another working directory still writes to the right place.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FORWARD_ROOT = PROJECT_ROOT / "erp_forward_operators"
INVERSE_ROOT = PROJECT_ROOT / "erp_inverse_operators"
INVERTIBLE_ROOT = PROJECT_ROOT / "erp_invertible_operators"

GENERAL = "GENERAL"
FREQ_HOLDOUT = "FREQ_HOLDOUT"
ALL_MODELS = "ALL_MODELS"


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------- forward -----------------------------------


def forward_model_path(model_short: str, dataset_tag: str, experiment: str = GENERAL) -> Path:
    return FORWARD_ROOT / "models" / experiment / dataset_tag / f"{model_short.lower()}.pth"


def forward_plot_root(dataset_tag: str, experiment: str = GENERAL) -> Path:
    """``plots/<EXPERIMENT>/<dataset>``; per-model folders live directly below."""
    return _mkdir(FORWARD_ROOT / "plots" / experiment / dataset_tag)


def forward_plot_dir(model_short: str, dataset_tag: str, experiment: str = GENERAL) -> Path:
    return _mkdir(forward_plot_root(dataset_tag, experiment) / model_short)


# ----------------------------- inverse -----------------------------------


def inverse_model_path(model_short: str, dataset_tag: str) -> Path:
    return INVERSE_ROOT / "models" / dataset_tag / f"inverse_{model_short.lower()}.pth"


def inverse_plot_dir(dataset_tag: str, model_short: str = ALL_MODELS) -> Path:
    return _mkdir(INVERSE_ROOT / "plots" / dataset_tag / model_short)


# ---------------------------- invertible ---------------------------------


def invertible_model_path(model_short: str, dataset_tag: str) -> Path:
    return INVERTIBLE_ROOT / "models" / dataset_tag / f"{model_short.lower()}.pth"


def invertible_plot_dir(dataset_tag: str, model_short: str = ALL_MODELS) -> Path:
    return _mkdir(INVERTIBLE_ROOT / "plots" / dataset_tag / model_short)


__all__ = [
    "PROJECT_ROOT",
    "GENERAL",
    "FREQ_HOLDOUT",
    "ALL_MODELS",
    "forward_model_path",
    "forward_plot_root",
    "forward_plot_dir",
    "inverse_model_path",
    "inverse_plot_dir",
    "invertible_model_path",
    "invertible_plot_dir",
]
