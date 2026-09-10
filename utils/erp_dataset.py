"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : ERP Dataset
Description : Shared ERP dataset pipeline for neural operators
==================================================

This module is intentionally independent of any neural-network architecture.
It generates and stores one common raw dataset using resonator features
``[m, k, f_t, x, y]`` and exposes it as one model-input view:

``multi_res``
    One ``[m, k, f_t, x, y]`` quintuplet per resonator, shape ``(num_res, 5)``.

The only target is ERP. ``m`` (mass) and ``f_t`` (tuning frequency) are the
two independently sampled primary quantities (LHS on both directly, so f_t
comes out uniform); resonator stiffness is derived from them:

    k = m * (2*pi*f_t)^2

Sampling f_t and k (or f_t and m via the old fixed-k scheme) directly instead
would make f_t -- the quantity whose distribution actually matters for
frequency-range coverage -- come out skewed, since f_t is a nonlinear
(square-root-of-ratio) function of any two independently sampled inputs.

Typical use from any neural-operator file
-----------------------------------------

    from erp_dataset import prepare_erp_dataset

    dataset, loaders = prepare_erp_dataset(
        num_samples=500,
        batch_size=64,
        dataset_file="datasets/dataset_erp_ft.pth",
        regenerate_dataset=False,
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

Normalization parameters, split IDs, and source-configuration IDs remain available
on ``dataset`` and can be saved with any model checkpoint.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from utils.physics import (
    Lx,
    Ly,
    freqs,
    m_max,
    m_min,
    num_res as default_num_res,
    resonator_bounds,
)
from utils.solver import compute_erp_spectrum
from utils.support import lhs_sampling, load_dataset, save_dataset


FEATURE_NAMES = ("m", "k", "f_t", "x", "y")
DATASET_SCHEMA_VERSION = 2
DEFAULT_DATASET_FILE = "datasets/dataset_erp_ft.pth"


# ==================================================
# Validation helpers
# ==================================================


def _copy_split_dict(splits: Mapping[str, Sequence[int]]) -> dict[str, np.ndarray]:
    required = {"train", "val", "test"}
    if set(splits) != required:
        raise ValueError("split_configuration_ids must contain exactly train/val/test.")
    return {
        name: np.asarray(splits[name], dtype=np.int64).copy()
        for name in ("train", "val", "test")
    }


# ==================================================
# Shared preprocessing / solver conversion helpers
# ==================================================


_CONFIG_ZSCORE_FIELDS = ("m", "k", "f_t", "x", "y")


def normalize_configuration_array(
    configuration: np.ndarray,
    norm_params: Mapping[str, object],
) -> np.ndarray:
    """Normalize one or more ``[..., num_res, 5]`` [m,k,f_t,x,y] configurations.

    Every field is independently z-scored (mean 0, std 1, statistics fit on
    the train split only) -- there is no bounded/range-scaled field anymore,
    since ``x``/``y`` no longer get the old ``x/Lx``, ``y/Ly`` treatment.
    """
    values = np.asarray(configuration, dtype=np.float32).copy()
    num_res = int(norm_params["num_res"])
    if values.shape[-2:] != (num_res, 5):
        raise ValueError(
            f"configuration must end with shape ({num_res}, 5) in [m,k,f_t,x,y] order."
        )
    for i, name in enumerate(_CONFIG_ZSCORE_FIELDS):
        values[..., i] = (
            values[..., i] - float(norm_params[f"{name}_mean"])
        ) / float(norm_params[f"{name}_std"])
    return values


def denormalize_configuration_array(
    configuration: np.ndarray,
    norm_params: Mapping[str, object],
) -> np.ndarray:
    """Inverse of :func:`normalize_configuration_array`; returns raw [m,k,f_t,x,y]."""
    values = np.asarray(configuration, dtype=np.float32).copy()
    num_res = int(norm_params["num_res"])
    if values.shape[-2:] != (num_res, 5):
        raise ValueError(
            f"configuration must end with shape ({num_res}, 5) in [m,k,f_t,x,y] order."
        )
    for i, name in enumerate(_CONFIG_ZSCORE_FIELDS):
        values[..., i] = (
            values[..., i] * float(norm_params[f"{name}_std"])
            + float(norm_params[f"{name}_mean"])
        )
    return values


def normalize_frequency_array(
    frequency: np.ndarray | float,
    norm_params: Mapping[str, object],
) -> np.ndarray:
    values = np.asarray(frequency, dtype=np.float32)
    return (
        (values - float(norm_params["freq_mean"]))
        / float(norm_params["freq_std"])
    ).astype(np.float32)


def normalize_erp_array(
    erp: np.ndarray | float,
    norm_params: Mapping[str, object],
) -> np.ndarray:
    values = np.asarray(erp, dtype=np.float32)
    return (
        (values - float(norm_params["erp_mean"]))
        / float(norm_params["erp_std"])
    ).astype(np.float32)


def denormalize_erp_array(
    normalized_erp: np.ndarray | torch.Tensor,
    norm_params: Mapping[str, object],
) -> np.ndarray:
    if torch.is_tensor(normalized_erp):
        values = normalized_erp.detach().cpu().numpy()
    else:
        values = np.asarray(normalized_erp)
    return (
        values.astype(np.float32) * float(norm_params["erp_std"])
        + float(norm_params["erp_mean"])
    )


def configuration_to_resonators(
    configuration: np.ndarray,
) -> list[dict[str, float]]:
    """Convert one raw ``(num_res, 5)`` [m,k,f_t,x,y] configuration for the solver.

    ``m`` and ``k`` are stored directly (independently sampled at generation
    time) rather than derived here -- ``f_t`` is carried along purely as an
    informational/model-input field, not used by the solver itself.
    """
    values = np.asarray(configuration, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("configuration must have shape (num_res, 5) in [m,k,f_t,x,y] order.")

    resonators: list[dict[str, float]] = []
    for m, k_i, f_t, x, y in values:
        resonators.append(
            {
                "f_t": float(f_t),
                "x": float(x),
                "y": float(y),
                "m": float(m),
                "c": 1.0,
                "k": float(k_i),
            }
        )
    return resonators


# ==================================================
# ERP dataset
# ==================================================


class ERPDataset(Dataset):
    """Shared configuration-frequency ERP dataset.

    Raw storage is architecture-independent:

    - ``configuration_features``: ``(n_configurations, num_res, 5)`` in ``[m, k, f_t, x, y]`` order
    - ``responses``: ``(n_configurations, n_freqs, 1)`` containing ERP in dB

    :meth:`__getitem__` returns the normalized configuration as one
    ``[m, k, f_t, x, y]`` quintuplet per resonator, shape ``(num_res, 5)``.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_res: int = default_num_res,
        seed: int = 727,
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if num_res <= 0:
            raise ValueError("num_res must be positive.")

        self.num_samples = int(num_samples)
        self.num_res = int(num_res)
        self.seed = int(seed)

        self.frequency_values = freqs.astype(np.float32, copy=True)
        self.configuration_features: np.ndarray | None = None
        self.responses: np.ndarray | None = None

        # IDs refer to rows in the original loaded/generated raw dataset.
        self.selected_source_ids: np.ndarray | None = None

        # IDs below are local to the currently selected configuration subset.
        self.split_configuration_ids: dict[str, np.ndarray] | None = None
        self.norm_params: dict[str, object] | None = None

    # --------------------------------------------------
    # Dataset generation / loading / saving
    # --------------------------------------------------

    @staticmethod
    def _sample_to_resonators(
        sample: np.ndarray,
        num_res: int,
    ) -> tuple[list[dict[str, float]], np.ndarray]:
        """Convert one LHS sample ``[x,y,f_t,m]*N`` to solver data and features.

        ``f_t`` and ``m`` are the two independently LHS-sampled primary
        quantities; stiffness ``k = m*(2*pi*f_t)**2`` is derived so that the
        resulting f_t distribution stays uniform (see module docstring).
        """
        quads = np.asarray(sample, dtype=np.float64).reshape(num_res, 4)
        x, y, f_t, m = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
        k = m * (2.0 * np.pi * f_t) ** 2
        features = np.stack([m, k, f_t, x, y], axis=1).astype(np.float32, copy=False)
        return configuration_to_resonators(features), features.copy()

    def generate(
        self,
        save: bool = True,
        filename: str | None = None,
        verbose: bool = True,
    ) -> dict[str, object]:
        """Generate ERP spectra for ``self.num_samples`` resonator configurations."""
        samples = lhs_sampling(
            self.num_samples,
            bounds=resonator_bounds(self.num_res),
            seed=self.seed,
        )

        n_freqs = self.frequency_values.size
        configuration_features = np.empty(
            (self.num_samples, self.num_res, 5), dtype=np.float32
        )
        responses = np.empty((self.num_samples, n_freqs, 1), dtype=np.float32)

        start = time.perf_counter()
        if verbose:
            print(
                f"Generating ERP dataset: {self.num_samples} configurations, "
                f"{n_freqs} frequencies/configuration"
            )
            print("Sampled variables : x, y, f_t, m  (independent LHS)")
            print("Stored features   : [m, k, f_t, x, y]")
            print(f"m range            : [{m_min}, {m_max}] kg")
            print("Derived stiffness  : k = m * (2*pi*f_t)^2")

        # Keep console output useful for large datasets without printing every row.
        report_every = max(1, self.num_samples // 20)

        for configuration_idx, sample in enumerate(samples):
            resonators, features = self._sample_to_resonators(sample, self.num_res)
            configuration_features[configuration_idx] = features
            responses[configuration_idx, :, 0] = compute_erp_spectrum(
                resonators,
                frequencies=self.frequency_values,
            ).astype(np.float32)

            if verbose and (
                (configuration_idx + 1) % report_every == 0
                or configuration_idx + 1 == self.num_samples
            ):
                elapsed = time.perf_counter() - start
                print(
                    f"Configuration {configuration_idx + 1}/{self.num_samples} complete "
                    f"(elapsed {elapsed:.2f} s)"
                )

        self.configuration_features = configuration_features
        self.responses = responses
        self.selected_source_ids = np.arange(self.num_samples, dtype=np.int64)
        self.split_configuration_ids = None
        self.norm_params = None

        payload = self.to_payload()
        if save:
            save_dataset(payload, filename or DEFAULT_DATASET_FILE)
        return payload

    def to_payload(self) -> dict[str, object]:
        """Return the architecture-independent raw dataset payload."""
        self._require_data()
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "target_name": "erp",
            "feature_names": list(FEATURE_NAMES),
            "feature_layout": "per_resonator_[m,k,f_t,x,y]",
            "num_samples": self.num_samples,
            "num_res": self.num_res,
            "seed": self.seed,
            "frequency_values": self.frequency_values,
            "configuration_features": self.configuration_features,
            "responses": self.responses,
            "plate_geometry": {"Lx": float(Lx), "Ly": float(Ly)},
            "resonator_mass_bounds": {"m_min": float(m_min), "m_max": float(m_max)},
        }

    def load(self, filename: str) -> "ERPDataset":
        """Load a raw ERP dataset created by this module."""
        payload = load_dataset(filename)

        feature_names = tuple(payload.get("feature_names", ()))
        if feature_names != FEATURE_NAMES:
            raise ValueError(
                f"{filename} does not contain the expected [m, k, f_t, x, y] dataset. "
                f"Found feature_names={feature_names or 'missing'}."
            )
        if str(payload.get("target_name", "")).lower() != "erp":
            raise ValueError(f"{filename} is not an ERP dataset.")

        self.num_samples = int(payload["num_samples"])
        self.num_res = int(payload["num_res"])
        self.seed = int(payload.get("seed", self.seed))
        self.frequency_values = np.asarray(
            payload["frequency_values"], dtype=np.float32
        )
        self.configuration_features = np.asarray(
            payload["configuration_features"], dtype=np.float32
        )
        self.responses = np.asarray(payload["responses"], dtype=np.float32)

        expected_configuration_shape = (self.num_samples, self.num_res, 5)
        expected_response_shape = (
            self.num_samples,
            self.frequency_values.size,
            1,
        )
        if self.configuration_features.shape != expected_configuration_shape:
            raise ValueError(
                f"Invalid configuration_features shape {self.configuration_features.shape}; "
                f"expected {expected_configuration_shape}."
            )
        if self.responses.shape != expected_response_shape:
            raise ValueError(
                f"Invalid responses shape {self.responses.shape}; "
                f"expected {expected_response_shape}."
            )

        self.selected_source_ids = np.arange(self.num_samples, dtype=np.int64)
        self.split_configuration_ids = None
        self.norm_params = None
        return self

    def load_shards(self, filenames: Sequence[str]) -> "ERPDataset":
        """Load and concatenate several same-schema dataset shards as one dataset.

        Some datasets (e.g. the 100k-configuration set) are stored as
        multiple files purely to stay under GitHub's 100MB per-file limit --
        each shard is a normal, independently loadable dataset saved by
        :meth:`generate`/:meth:`to_payload`, just covering a different slice
        of configurations. This concatenates them in memory into one
        dataset with no separate merged file ever needing to exist on disk.
        """
        if not filenames:
            raise ValueError("filenames must not be empty.")

        payloads = [load_dataset(f) for f in filenames]
        first = payloads[0]

        feature_names = tuple(first.get("feature_names", ()))
        if feature_names != FEATURE_NAMES:
            raise ValueError(
                f"{filenames[0]} does not contain the expected [m, k, f_t, x, y] dataset. "
                f"Found feature_names={feature_names or 'missing'}."
            )
        num_res = int(first["num_res"])
        frequency_values = np.asarray(first["frequency_values"], dtype=np.float32)
        for filename, payload in zip(filenames[1:], payloads[1:], strict=True):
            if int(payload["num_res"]) != num_res:
                raise ValueError(f"{filename} has a different num_res than {filenames[0]}.")
            if not np.array_equal(
                np.asarray(payload["frequency_values"], dtype=np.float32), frequency_values
            ):
                raise ValueError(f"{filename} has different frequency_values than {filenames[0]}.")

        self.num_res = num_res
        self.seed = int(first.get("seed", self.seed))
        self.frequency_values = frequency_values
        self.configuration_features = np.concatenate(
            [np.asarray(p["configuration_features"], dtype=np.float32) for p in payloads], axis=0
        )
        self.responses = np.concatenate(
            [np.asarray(p["responses"], dtype=np.float32) for p in payloads], axis=0
        )
        self.num_samples = self.configuration_features.shape[0]

        self.selected_source_ids = np.arange(self.num_samples, dtype=np.int64)
        self.split_configuration_ids = None
        self.norm_params = None
        return self

    # --------------------------------------------------
    # Configuration subset and split
    # --------------------------------------------------

    def select_configurations(
        self,
        num_configurations: int | None = None,
        seed: int | None = None,
        source_ids: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Select the configuration subset used by a model experiment.

        ``num_configurations`` is applied *after loading* the raw dataset.  Supplying
        ``source_ids`` restores an exact previously used subset.
        """
        self._require_data()
        total = int(self.configuration_features.shape[0])

        if source_ids is not None:
            selected = np.asarray(source_ids, dtype=np.int64)
            if selected.ndim != 1 or selected.size < 3:
                raise ValueError("source_ids must contain at least 3 configuration indices.")
            if np.unique(selected).size != selected.size:
                raise ValueError("source_ids must not contain duplicates.")
            if np.any(selected < 0) or np.any(selected >= total):
                raise ValueError("source_ids contain indices outside the raw dataset.")
        else:
            num_configurations = total if num_configurations is None else int(num_configurations)
            if num_configurations < 3:
                raise ValueError("At least 3 configurations are required.")
            if num_configurations > total:
                raise ValueError(
                    f"Requested {num_configurations} configurations, but the dataset contains "
                    f"only {total}."
                )

            if num_configurations == total:
                selected = np.arange(total, dtype=np.int64)
            else:
                rng = np.random.default_rng(self.seed if seed is None else seed)
                selected = np.sort(
                    rng.choice(total, size=num_configurations, replace=False).astype(np.int64)
                )

        self.configuration_features = self.configuration_features[selected].copy()
        self.responses = self.responses[selected].copy()
        self.num_samples = int(selected.size)
        self.selected_source_ids = selected.copy()
        self.split_configuration_ids = None
        self.norm_params = None
        return selected

    def split_configurations(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int | None = None,
        split_configuration_ids: Mapping[str, Sequence[int]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Split by complete resonator configuration, never by individual frequency."""
        self._require_data()

        if split_configuration_ids is not None:
            splits = _copy_split_dict(split_configuration_ids)
            all_ids = np.concatenate(list(splits.values()))
            if all_ids.size != self.num_samples:
                raise ValueError(
                    "Restored split IDs must cover every selected configuration exactly once."
                )
            if np.unique(all_ids).size != self.num_samples:
                raise ValueError("Restored split IDs overlap or contain duplicates.")
            if np.any(all_ids < 0) or np.any(all_ids >= self.num_samples):
                raise ValueError("Restored split IDs are outside the selected subset.")
            if any(splits[name].size == 0 for name in splits):
                raise ValueError("train, val, and test splits must all be non-empty.")
            self.split_configuration_ids = splits
            return splits

        if self.num_samples < 3:
            raise ValueError("At least 3 configurations are required for train/val/test splits.")
        if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0):
            raise ValueError("train_ratio and val_ratio must both lie in (0, 1).")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.")

        rng = np.random.default_rng(self.seed if seed is None else seed)
        configuration_ids = rng.permutation(self.num_samples)

        n_train = max(1, int(np.floor(train_ratio * self.num_samples)))
        n_val = max(1, int(np.floor(val_ratio * self.num_samples)))
        if n_train + n_val >= self.num_samples:
            n_train = self.num_samples - 2
            n_val = 1

        self.split_configuration_ids = {
            "train": configuration_ids[:n_train],
            "val": configuration_ids[n_train : n_train + n_val],
            "test": configuration_ids[n_train + n_val :],
        }
        return self.split_configuration_ids

    # --------------------------------------------------
    # Normalization
    # --------------------------------------------------

    def fit_normalization(
        self,
        train_configuration_ids: Sequence[int] | None = None,
    ) -> dict[str, object]:
        """Fit data-dependent scaling using training configurations only.

        Every resonator field (``m``, ``k``, ``f_t``, ``x``, ``y``) uses one
        shared mean/std across all resonators and is z-scored (mean 0, std 1)
        -- there is no bounded/range-scaled field.  This avoids artificial
        resonator-slot-specific scaling and makes the preprocessing more
        suitable for set/graph/neural-operator architectures.
        """
        self._require_data()

        if train_configuration_ids is None:
            if self.split_configuration_ids is None:
                self.split_configurations()
            train_ids = self.split_configuration_ids["train"]
        else:
            train_ids = np.asarray(train_configuration_ids, dtype=np.int64)

        train_features = self.configuration_features[train_ids]
        train_erp = self.responses[train_ids, :, 0]

        self.norm_params = {
            "feature_names": list(FEATURE_NAMES),
            "num_res": self.num_res,
        }
        for i, name in enumerate(_CONFIG_ZSCORE_FIELDS):
            values = train_features[:, :, i]
            self.norm_params[f"{name}_mean"] = np.float32(values.mean())
            self.norm_params[f"{name}_std"] = np.float32(max(float(values.std()), 1e-8))

        # The frequency grid is common to every configuration, so this does not use
        # target information from validation/test configurations.
        self.norm_params["freq_mean"] = np.float32(self.frequency_values.mean())
        self.norm_params["freq_std"] = np.float32(
            max(float(self.frequency_values.std()), 1e-8)
        )

        self.norm_params["erp_mean"] = np.float32(train_erp.mean())
        self.norm_params["erp_std"] = np.float32(max(float(train_erp.std()), 1e-12))
        return self.norm_params

    def set_normalization(self, norm_params: Mapping[str, object]) -> None:
        """Restore normalization previously saved with a trained model."""
        required = {"feature_names", "num_res", "freq_mean", "freq_std", "erp_mean", "erp_std"}
        for name in _CONFIG_ZSCORE_FIELDS:
            required.add(f"{name}_mean")
            required.add(f"{name}_std")
        missing = required.difference(norm_params)
        if missing:
            raise ValueError(f"Normalization state is missing: {sorted(missing)}")
        if tuple(norm_params["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Normalization state does not use [m, k, f_t, x, y] features.")
        if int(norm_params["num_res"]) != self.num_res:
            raise ValueError("Normalization num_res does not match the dataset.")
        self.norm_params = dict(norm_params)

    def normalize_configuration(self, configuration: np.ndarray) -> np.ndarray:
        """Normalize one or more ``[..., num_res, 5]`` [m,k,f_t,x,y] configurations."""
        self._require_normalization()
        return normalize_configuration_array(configuration, self.norm_params)

    def normalize_frequency(self, frequency: np.ndarray | float) -> np.ndarray:
        """Normalize one or more forcing/evaluation frequencies."""
        self._require_normalization()
        return normalize_frequency_array(frequency, self.norm_params)

    def normalize_erp(self, erp: np.ndarray | float) -> np.ndarray:
        self._require_normalization()
        return normalize_erp_array(erp, self.norm_params)

    def denormalize_erp(self, normalized_erp: np.ndarray | torch.Tensor) -> np.ndarray:
        """Convert normalized ERP predictions back to physical ERP/dB values."""
        self._require_normalization()
        return denormalize_erp_array(normalized_erp, self.norm_params)

    # --------------------------------------------------
    # Input views / DataLoaders
    # --------------------------------------------------

    @property
    def model_input_shape(self) -> tuple[int, ...]:
        return (self.num_res, 5)

    def dataloaders(
        self,
        batch_size: int = 64,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int | None = None,
        num_workers: int = 0,
        pin_memory: bool | None = None,
        split_configuration_ids: Mapping[str, Sequence[int]] | None = None,
        norm_params: Mapping[str, object] | None = None,
    ) -> dict[str, DataLoader]:
        """Create normalized train/validation/test DataLoaders."""
        self._require_data()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        splits = self.split_configurations(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            split_configuration_ids=split_configuration_ids,
        )

        if norm_params is None:
            self.fit_normalization(splits["train"])
        else:
            self.set_normalization(norm_params)

        n_freqs = self.frequency_values.size

        def sample_indices(configuration_ids: np.ndarray) -> np.ndarray:
            return (
                configuration_ids[:, None] * n_freqs
                + np.arange(n_freqs, dtype=np.int64)[None, :]
            ).reshape(-1)

        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        loader_seed = self.seed if seed is None else int(seed)
        loaders: dict[str, DataLoader] = {}
        for split_name in ("train", "val", "test"):
            subset = Subset(self, sample_indices(splits[split_name]).tolist())

            # A dedicated generator makes training shuffling reproducible.
            generator = None
            if split_name == "train":
                generator = torch.Generator()
                generator.manual_seed(loader_seed)

            loaders[split_name] = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
                generator=generator,
            )

        return loaders

    # --------------------------------------------------
    # Reproducibility state for model checkpoints
    # --------------------------------------------------

    def preprocessing_state(self) -> dict[str, object]:
        """State to save with any neural-operator checkpoint.

        Restoring this state reproduces the exact raw-configuration subset, configuration-level
        split, and normalization used during training.
        """
        if self.selected_source_ids is None:
            raise RuntimeError("No configuration subset has been selected.")
        if self.split_configuration_ids is None:
            raise RuntimeError("Dataset has not been split.")
        self._require_normalization()

        return {
            "selected_source_ids": self.selected_source_ids.copy(),
            "split_configuration_ids": {
                name: ids.copy() for name, ids in self.split_configuration_ids.items()
            },
            "norm_params": dict(self.norm_params),
            "num_res": self.num_res,
            "feature_names": list(FEATURE_NAMES),
            # The raw file(s) selected_source_ids indexes into. Absent (None,
            # via .get() at the read site) on checkpoints saved before this
            # field existed -- those fall back to whatever dataset_file the
            # caller passes, same as always.
            "dataset_file": getattr(self, "raw_dataset_file", None),
        }

    # --------------------------------------------------
    # torch Dataset API
    # --------------------------------------------------

    def __len__(self) -> int:
        if self.responses is None:
            return 0
        return self.num_samples * self.frequency_values.size

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_data()
        self._require_normalization()

        n_freqs = self.frequency_values.size
        configuration_idx, freq_idx = divmod(int(index), n_freqs)

        # normalize_configuration_array() already returns float32 (num_res, 5).
        configuration_input = self.normalize_configuration(self.configuration_features[configuration_idx])
        frequency = np.asarray(
            [self.normalize_frequency(self.frequency_values[freq_idx])],
            dtype=np.float32,
        ).reshape(1)
        erp = np.asarray(
            [self.normalize_erp(self.responses[configuration_idx, freq_idx, 0])],
            dtype=np.float32,
        ).reshape(1)

        return (
            torch.from_numpy(configuration_input),
            torch.from_numpy(frequency),
            torch.from_numpy(erp),
        )

    # --------------------------------------------------
    # Internal checks / summary
    # --------------------------------------------------

    def _require_data(self) -> None:
        if self.configuration_features is None or self.responses is None:
            raise RuntimeError("Generate or load the ERP dataset before using it.")

    def _require_normalization(self) -> None:
        if self.norm_params is None:
            raise RuntimeError(
                "Normalization is not available. Call dataloaders() or "
                "fit_normalization() first."
            )

    def summary(self) -> dict[str, object]:
        """Return compact dataset information useful for experiment logging."""
        self._require_data()
        return {
            "num_configurations": self.num_samples,
            "num_res": self.num_res,
            "num_frequencies": int(self.frequency_values.size),
            "frequency_range_hz": (
                float(self.frequency_values.min()),
                float(self.frequency_values.max()),
            ),
            "feature_names": list(FEATURE_NAMES),
            "model_input_shape": self.model_input_shape,
            "target_shape": (1,),
            "total_configuration_frequency_samples": len(self),
            "split_sizes_configurations": (
                None
                if self.split_configuration_ids is None
                else {
                    name: int(ids.size)
                    for name, ids in self.split_configuration_ids.items()
                }
            ),
        }


# ==================================================
# One-function public workflow
# ==================================================


def prepare_erp_dataset(
    num_samples: int = 100,
    *,
    batch_size: int = 64,
    num_res: int = default_num_res,
    dataset_file: str | Sequence[str] = DEFAULT_DATASET_FILE,
    regenerate_dataset: bool = False,
    num_generate: int | None = None,
    seed: int = 727,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    preprocessing_state: Mapping[str, object] | None = None,
    verbose: bool = True,
) -> tuple[ERPDataset, dict[str, DataLoader]]:
    """Load/generate, subset, split, normalize, and build ERP DataLoaders.

    This is the only function neural-operator files normally need to call.

    Parameters
    ----------
    num_samples:
        Number of configurations to use *after loading* the raw dataset.
    batch_size:
        Number of configuration-frequency pairs per mini-batch.
    num_res:
        Number of resonators per configuration.
    dataset_file:
        Shared raw ERP dataset file. Pass a list/tuple of filenames instead
        of one string to load a dataset stored as multiple same-schema
        shards (e.g. a dataset split to stay under a hosting size limit) --
        see :meth:`ERPDataset.load_shards`. Sharded datasets are loaded
        as-is and never auto-generated/regenerated here.
    regenerate_dataset:
        If True, regenerate and overwrite ``dataset_file``. Not supported
        when ``dataset_file`` is a list of shards.
    num_generate:
        Number of raw configurations to generate when creating/regenerating the file.
        If omitted, ``num_samples`` configurations are generated.
        Example: generate 5000 once but train with only 500 by setting
        ``num_generate=5000, num_samples=500``.
    seed:
        LHS generation, subset selection, configuration split, and DataLoader seed.
    train_ratio, val_ratio:
        Configuration-level split ratios. Test gets the remainder.
    preprocessing_state:
        Optional state returned by ``dataset.preprocessing_state()``.  Passing
        it restores the exact source-configuration subset, split, and normalization,
        which is useful for model evaluation and fair operator comparisons.
    """
    num_samples = int(num_samples)
    if num_samples < 3:
        raise ValueError("num_samples must be at least 3.")

    is_sharded = not isinstance(dataset_file, (str, Path))

    if is_sharded:
        if regenerate_dataset:
            raise ValueError("regenerate_dataset is not supported with a sharded dataset_file.")
        if verbose:
            print(f"Loading sharded raw ERP dataset ({len(dataset_file)} files): {list(dataset_file)}")
        dataset = ERPDataset(num_samples=max(num_samples, 3), num_res=num_res, seed=seed)
        dataset.load_shards(list(dataset_file))

        if dataset.num_res != int(num_res):
            raise ValueError(
                f"Loaded dataset has num_res={dataset.num_res}, but num_res={num_res} "
                "was requested. Use a matching dataset file or regenerate it."
            )
    else:
        path = Path(dataset_file)
        should_generate = regenerate_dataset or not path.exists()

        if should_generate:
            n_generate = num_samples if num_generate is None else int(num_generate)
            if n_generate < 3:
                raise ValueError("num_generate must be at least 3.")
            if n_generate < num_samples and preprocessing_state is None:
                raise ValueError("num_generate cannot be smaller than num_samples.")

            if verbose:
                reason = "regenerate_dataset=True" if regenerate_dataset else "file missing"
                print(f"Creating raw ERP dataset ({reason}): {dataset_file}")

            dataset = ERPDataset(
                num_samples=n_generate,
                num_res=num_res,
                seed=seed,
            )
            dataset.generate(save=True, filename=dataset_file, verbose=verbose)
        else:
            dataset = ERPDataset(
                num_samples=max(num_samples, 3),
                num_res=num_res,
                seed=seed,
            )
            dataset.load(dataset_file)

            if dataset.num_res != int(num_res):
                raise ValueError(
                    f"Loaded dataset has num_res={dataset.num_res}, but num_res={num_res} "
                    "was requested. Use a matching dataset file or regenerate it."
                )

    # Recorded so a saved checkpoint's preprocessing_state can remember which
    # raw file its selected_source_ids index into -- different raw dataset
    # files (e.g. the 10k vs. the 100k sharded dataset) are independently
    # sampled, so the same numeric index means a different physical
    # configuration in each one. Without this, evaluating/predicting with a
    # checkpoint against the wrong raw file fails with a confusing
    # "indices outside the raw dataset" error instead of using the file the
    # checkpoint actually needs.
    dataset.raw_dataset_file = list(dataset_file) if is_sharded else str(dataset_file)

    # Restore an exact trained preprocessing state when requested; otherwise
    # choose num_samples configurations reproducibly from the raw dataset.
    if preprocessing_state is not None:
        if tuple(preprocessing_state.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("preprocessing_state does not use [m, k, f_t, x, y] features.")
        if int(preprocessing_state.get("num_res", -1)) != dataset.num_res:
            raise ValueError("preprocessing_state num_res does not match the dataset.")

        source_ids = np.asarray(
            preprocessing_state["selected_source_ids"], dtype=np.int64
        )
        dataset.select_configurations(source_ids=source_ids)
        restored_splits = preprocessing_state["split_configuration_ids"]
        restored_norm = preprocessing_state["norm_params"]
    else:
        dataset.select_configurations(num_configurations=num_samples, seed=seed)
        restored_splits = None
        restored_norm = None

    loaders = dataset.dataloaders(
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
        split_configuration_ids=restored_splits,
        norm_params=restored_norm,
    )

    if verbose:
        info = dataset.summary()
        split_sizes = info["split_sizes_configurations"]
        raw_file_display = ", ".join(dataset_file) if is_sharded else str(dataset_file)
        print("=" * 68)
        print("ERP dataset ready")
        print(f"Raw file              : {raw_file_display}")
        print(f"Configurations used          : {info['num_configurations']}")
        print(f"Resonators/configuration     : {info['num_res']}")
        print(f"Frequencies/configuration    : {info['num_frequencies']}")
        print(f"Model input shape     : {info['model_input_shape']}")
        print(f"Target                : ERP, shape {info['target_shape']}")
        print(
            "Configuration split          : "
            f"{split_sizes['train']} train / "
            f"{split_sizes['val']} val / "
            f"{split_sizes['test']} test"
        )
        print(f"Batch size            : {batch_size}")
        print("=" * 68)

    return dataset, loaders


__all__ = [
    "ERPDataset",
    "prepare_erp_dataset",
    "normalize_configuration_array",
    "denormalize_configuration_array",
    "normalize_frequency_array",
    "normalize_erp_array",
    "denormalize_erp_array",
    "configuration_to_resonators",
    "FEATURE_NAMES",
    "DEFAULT_DATASET_FILE",
]


# ==================================================
# Initial raw-dataset generation
# ==================================================

if __name__ == "__main__":
    # Initial master-dataset generation.
    NUM_CONFIGURATIONS_TO_GENERATE = 10000

    dataset = ERPDataset(
        num_samples=NUM_CONFIGURATIONS_TO_GENERATE,
        num_res=default_num_res,
        seed=727,
    )
    dataset.generate(
        save=True,
        filename=DEFAULT_DATASET_FILE,
        verbose=True,
    )
    print(f"Dataset size: {Path('datasets/dataset_erp_ft.pth').stat().st_size / (1024**2):.2f} MB")
    data = torch.load("datasets/dataset_erp_ft.pth", map_location="cpu", weights_only=False)
    print("Configurations:", data["num_samples"])
    print("Frequencies per configuration:", len(data["frequency_values"]))
    print("Total samples:", data["num_samples"] * len(data["frequency_values"]))
