"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Paths
Description : One place that decides where every checkpoint and plot goes
==================================================

Every model family lives in its own top-level folder with the same three
subfolders -- ``scripts/`` (code), ``models/`` (checkpoints), ``plots/``
(figures) -- and plots are sorted by type::

    erp_forward/                    forward operators, design -> ERP
        models/<dataset>/<model>.pth
        models/experiments/<experiment>/<dataset>/...   (frequency_holdout, dco_variants)
        models/legacy/
        plots/models/<dataset>/<MODEL>/  and  plots/models/<dataset>/ALL_MODELS/
        plots/experiments/<experiment>/...  plots/architectures/  plots/legacy/
    erp_inverse/                    inverse models, ERP -> design
        models/<dataset>/inverse_<model>.pth
        models/fixed_resonator/<dataset>/<model>.pth   (position-only models, model bank blocks)
        plots/models/<dataset>/<MODEL>/ and .../ALL_MODELS/
        plots/fixed_resonator/<dataset>/<MODEL>/
        plots/experiments/block_bank[_3res]/
    erp_invertible/                 coupling-flow invertible operators (iFNO, iDCO, ...)
        models/<dataset>/<model>.pth
        plots/models/<dataset>/<MODEL>/ and .../ALL_MODELS/   plots/architectures/
    erp_invertible_deeponet/        invertible DeepONet (Kaltenbach et al.)
        models/<dataset>/idon_<variant>.pth
        plots/models/<dataset>/<VARIANT>/ and .../ALL_MODELS/  plots/architectures/
    disp_forward/                   forward operators, design -> displacement field
        models/<model>.pth
        plots/models/<MODEL>/

    datasets/                       raw data, sorted by type
        erp/3res/<tag>/  erp/2res/<tag>/  erp/blocks_2res/  erp/blocks_3res/
        displacement/    scripts/ (generators)

All paths are absolute (anchored at the repository root), so running
``main.py`` from another working directory still writes to the right place.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where models/ and plots/ are written. Defaults to the repository itself;
# set THESIS_OUTPUT_ROOT (e.g. to a scratch or Google Drive folder) to send
# every checkpoint and figure there instead -- the family folders are then
# created below it with the same layout. Datasets are always read from the
# repository.
OUTPUT_ROOT = Path(os.environ.get("THESIS_OUTPUT_ROOT") or PROJECT_ROOT).resolve()

FORWARD_ROOT = OUTPUT_ROOT / "erp_forward"
INVERSE_ROOT = OUTPUT_ROOT / "erp_inverse"
INVERTIBLE_ROOT = OUTPUT_ROOT / "erp_invertible"
IDON_ROOT = OUTPUT_ROOT / "erp_invertible_deeponet"
DISPLACEMENT_ROOT = OUTPUT_ROOT / "disp_forward"
DATASETS_ROOT = PROJECT_ROOT / "datasets"

GENERAL = "GENERAL"
FREQ_HOLDOUT = "FREQ_HOLDOUT"
ALL_MODELS = "ALL_MODELS"

# experiment name -> folder below models/experiments and plots/experiments
_EXPERIMENT_DIRS = {FREQ_HOLDOUT: "frequency_holdout"}


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_path(path: str | Path) -> Path:
    """Absolute path for a repository-relative path (absolute paths unchanged)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_existing(path: str | Path) -> Path:
    """``path`` if it exists; otherwise, when outputs are redirected
    (THESIS_OUTPUT_ROOT), the same file inside the repository -- so trained
    models shipped with the repository stay usable (e.g. the frozen forward
    surrogates). Returns ``path`` unchanged if neither exists."""
    path = Path(path)
    if path.exists() or OUTPUT_ROOT == PROJECT_ROOT:
        return path
    try:
        fallback = PROJECT_ROOT / path.resolve().relative_to(OUTPUT_ROOT)
    except ValueError:
        return path
    return fallback if fallback.exists() else path


def architecture_dir(root: Path) -> Path:
    """``<family>/plots/architectures`` (diagrams of the models)."""
    return _mkdir(root / "plots" / "architectures")


# ----------------------------- forward -----------------------------------


def _forward_sub(kind: str, dataset_tag: str, experiment: str) -> Path:
    if experiment == GENERAL:  # models/<dataset> and plots/models/<dataset>
        return FORWARD_ROOT / "models" / dataset_tag if kind == "models" else FORWARD_ROOT / "plots" / "models" / dataset_tag
    folder = _EXPERIMENT_DIRS.get(experiment, experiment.lower())
    return FORWARD_ROOT / kind / "experiments" / folder / dataset_tag


def forward_model_path(model_short: str, dataset_tag: str, experiment: str = GENERAL) -> Path:
    return _forward_sub("models", dataset_tag, experiment) / f"{model_short.lower()}.pth"


def forward_plot_root(dataset_tag: str, experiment: str = GENERAL) -> Path:
    """``plots/models/<dataset>`` (or ``plots/experiments/<experiment>/<dataset>``);
    per-model folders live directly below."""
    return _mkdir(_forward_sub("plots", dataset_tag, experiment))


def forward_plot_dir(model_short: str, dataset_tag: str, experiment: str = GENERAL) -> Path:
    return _mkdir(forward_plot_root(dataset_tag, experiment) / model_short)


# ----------------------------- inverse -----------------------------------


def inverse_model_path(model_short: str, dataset_tag: str) -> Path:
    return INVERSE_ROOT / "models" / dataset_tag / f"inverse_{model_short.lower()}.pth"


def inverse_plot_dir(dataset_tag: str, model_short: str = ALL_MODELS) -> Path:
    return _mkdir(INVERSE_ROOT / "plots" / "models" / dataset_tag / model_short)


def fixed_resonator_model_path(model_short: str, dataset_tag: str) -> Path:
    """Position-only inverse models (erp_inverse/scripts/fixed_resonator)."""
    return INVERSE_ROOT / "models" / "fixed_resonator" / dataset_tag / f"{model_short.lower()}.pth"


def fixed_resonator_plot_dir(dataset_tag: str, model_short: str = ALL_MODELS) -> Path:
    return _mkdir(INVERSE_ROOT / "plots" / "fixed_resonator" / dataset_tag / model_short)


def block_bank_plot_dir(num_res: int = 2) -> Path:
    name = "block_bank" if int(num_res) == 2 else f"block_bank_{int(num_res)}res"
    return _mkdir(INVERSE_ROOT / "plots" / "experiments" / name)


# ---------------------------- invertible ---------------------------------


def invertible_model_path(model_short: str, dataset_tag: str) -> Path:
    return INVERTIBLE_ROOT / "models" / dataset_tag / f"{model_short.lower()}.pth"


def invertible_plot_dir(dataset_tag: str, model_short: str = ALL_MODELS) -> Path:
    return _mkdir(INVERTIBLE_ROOT / "plots" / "models" / dataset_tag / model_short)


# ------------------------- invertible DeepONet -----------------------------


def idon_model_path(variant: str, dataset_tag: str) -> Path:
    return IDON_ROOT / "models" / dataset_tag / f"idon_{variant.lower()}.pth"


def idon_plot_dir(dataset_tag: str, variant: str = ALL_MODELS) -> Path:
    return _mkdir(IDON_ROOT / "plots" / "models" / dataset_tag / variant)


# --------------------------- displacement ---------------------------------


def displacement_model_path(model_short: str) -> Path:
    return DISPLACEMENT_ROOT / "models" / f"{model_short.lower()}.pth"


def displacement_plot_dir(model_short: str) -> Path:
    return _mkdir(DISPLACEMENT_ROOT / "plots" / "models" / model_short)


__all__ = [
    "PROJECT_ROOT", "OUTPUT_ROOT", "FORWARD_ROOT", "INVERSE_ROOT", "INVERTIBLE_ROOT", "IDON_ROOT", "DISPLACEMENT_ROOT",
    "DATASETS_ROOT", "GENERAL", "FREQ_HOLDOUT", "ALL_MODELS",
    "project_path", "find_existing", "architecture_dir",
    "forward_model_path", "forward_plot_root", "forward_plot_dir",
    "inverse_model_path", "inverse_plot_dir", "fixed_resonator_model_path", "fixed_resonator_plot_dir",
    "block_bank_plot_dir",
    "invertible_model_path", "invertible_plot_dir",
    "idon_model_path", "idon_plot_dir",
    "displacement_model_path", "displacement_plot_dir",
]
