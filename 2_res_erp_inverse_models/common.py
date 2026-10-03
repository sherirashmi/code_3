"""Shared data pipeline, paths and checkpoints of the position-only inverse models.

Differences to ``erp_inverse_operators`` (everything else -- spectrum encoder,
training loop, solver scoring -- is the same machinery):

* Design: only the resonator positions, ``2 * num_res`` numbers in the bounded
  logit space of ``design_space.py``. The dataset must have identical, fixed
  resonators (``fixed_resonator`` in the registry, e.g.
  ``100k_2res_fixed_m0.2_ft72_18modes``); their m, f_t and k are stored in
  ``norm_params`` and re-attached to every predicted design for the solver.
* Canonical order: resonators sorted by ascending x (all f_t are equal).
* Input: only the ERP inside ``[F_LOW, F_HIGH] = [40, 120] Hz`` (161 of the 301
  frequency points), where the ERP depends on the positions; below 40 Hz the
  ERP is the same for every configuration and above 120 Hz it hardly varies
  (see ``dataset_analysis/100k_2res_fixed_m0.2_ft72_18modes/plots/erp_frequency_boxplot.png``).

Models, checkpoints and plots live inside this folder:
``models/<dataset>/<model>.pth`` and ``plots/<dataset>/<MODEL>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from erp_forward_operators.neural_operator_utils import _configuration_features, _split_ids, prepare_operator_data
from utils.erp_dataset import DATASETS, normalize_erp_array, select_dataset_modal_resolution
from utils.support import device

from .design_space import encode_positions, fit_position_stats, sort_by_x

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = "100k_2res_fixed_m0.2_ft72_18modes"
F_LOW, F_HIGH = 40.0, 120.0  # Hz, ERP band the models see
ALL_MODELS = "ALL_MODELS"


def model_path(name: str, dataset_tag: str) -> Path:
    return PACKAGE_ROOT / "models" / dataset_tag / f"{name.lower()}.pth"


def plot_dir(dataset_tag: str, name: str = ALL_MODELS) -> Path:
    path = PACKAGE_ROOT / "plots" / dataset_tag / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def band_mask(frequency_values: np.ndarray, f_low: float = F_LOW, f_high: float = F_HIGH) -> np.ndarray:
    freq = np.asarray(frequency_values, dtype=np.float64)
    return (freq >= f_low - 1e-9) & (freq <= f_high + 1e-9)


class PositionInverseDataset(Dataset):
    """One item = (normalised ERP in the band, normalised positions ``(num_res, 2)`` sorted by x)."""

    def __init__(self, dataset, configuration_ids: np.ndarray, mask: np.ndarray) -> None:
        ids = np.asarray(configuration_ids, dtype=np.int64)
        norm = dataset.norm_params
        raw = sort_by_x(_configuration_features(dataset)[ids])
        response = normalize_erp_array(np.asarray(dataset.responses, dtype=np.float32)[ids, :, 0], norm)[:, mask]
        self.configuration = torch.from_numpy(encode_positions(raw, norm))
        self.spectrum = torch.from_numpy(np.ascontiguousarray(response, dtype=np.float32))

    def __len__(self) -> int:
        return self.configuration.shape[0]

    def __getitem__(self, index: int):
        return self.spectrum[index], self.configuration[index]


def _fixed_resonator(raw: np.ndarray) -> tuple[float, float]:
    """The common (m, f_t) of every resonator; refuses datasets where they vary."""
    m, f_t = raw[..., 0], raw[..., 2]
    if not (np.allclose(m, m.flat[0], rtol=1e-5) and np.allclose(f_t, f_t.flat[0], rtol=1e-5)):
        raise ValueError(
            "Position-only inverse models need a dataset of identical, fixed resonators "
            f"(e.g. '{DEFAULT_DATASET}'); this dataset varies m and/or f_t."
        )
    return float(m.flat[0]), float(f_t.flat[0])


def prepare_data(
    dataset_tag: str = DEFAULT_DATASET,
    num_configurations: int | None = None,
    batch_size: int = 64,
    seed: int = 727,
    f_low: float = F_LOW,
    f_high: float = F_HIGH,
):
    """Train / val / test loaders (same 80/10/10 split as every other model in
    the project) of ``(ERP in [f_low, f_high], positions)`` pairs.

    ``dataset.norm_params`` gains the position statistics (training split), the
    fixed m / f_t, the band limits and ``n_freq_band``, so every checkpoint
    carries everything needed to decode and evaluate its predictions.
    """
    spec = DATASETS[dataset_tag]
    select_dataset_modal_resolution(dataset_tag)
    files = list(spec["files"])
    dataset, _ = prepare_operator_data(
        num_configurations=int(num_configurations or spec["num_configurations"]),
        batch_size=16,
        dataset_file=files if len(files) > 1 else files[0],
        regenerate_dataset=False,
        seed=seed,
    )
    splits = _split_ids(dataset)
    raw = _configuration_features(dataset)
    fixed_m, fixed_f_t = _fixed_resonator(raw)
    mask = band_mask(dataset.frequency_values, f_low, f_high)
    dataset.norm_params.update(fit_position_stats(sort_by_x(raw[splits["train"]])))
    dataset.norm_params.update({
        "fixed_m": fixed_m, "fixed_f_t": fixed_f_t, "f_low": float(f_low), "f_high": float(f_high),
        "n_freq_band": int(mask.sum()), "num_res": int(dataset.num_res),
    })
    dataset.band_mask = mask
    dataset.band_frequency_values = np.asarray(dataset.frequency_values)[mask]
    print(f"Inverse data: ERP band {f_low:g}-{f_high:g} Hz ({int(mask.sum())} of {mask.size} frequency points), "
          f"design = {2 * dataset.num_res} positions (sorted by x), fixed m = {fixed_m:g} kg, f_t = {fixed_f_t:g} Hz")
    pin_memory = device.type == "cuda"
    loaders = {
        name: DataLoader(PositionInverseDataset(dataset, ids, mask), batch_size=batch_size,
                         shuffle=(name == "train"), pin_memory=pin_memory)
        for name, ids in splits.items()
    }
    return dataset, loaders


def save_checkpoint(model: torch.nn.Module, norm_params: Mapping[str, object], path: Path,
                    extra: Mapping[str, object] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "norm_params": dict(norm_params)}
    payload.update(dict(extra or {}))
    torch.save(payload, path)
    print(f"Checkpoint saved: {path}")
