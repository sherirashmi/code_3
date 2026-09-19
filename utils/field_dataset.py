"""Field-level displacement dataset: complex plate displacement w(x,y,omega)
at randomly sampled collocation points, for each of many LHS-sampled
resonator configurations.

Unlike the ERP-only dataset (utils/erp_dataset.py), which stores one
already-spatially-integrated scalar per frequency
(ERP = rho*c*integral(|w_dot|^2) dA), this stores the underlying complex
displacement FIELD itself at a modest number of spatial points per
configuration -- the quantity a genuine PDE-residual physics-informed
loss on the Kirchhoff-Love plate equation would actually need, since you
cannot take spatial derivatives of an already-integrated scalar.

Storage format: reuses this project's existing torch.save/.load
convention (utils.support.save_dataset/load_dataset) instead of
introducing a new dependency such as pandas/pyarrow (Parquet) or h5py
(HDF5). This repo has no dependency manifest at all, every other dataset
already uses this same mechanism, and at the scale generated here (a few
million field samples, dense arrays) everything fits comfortably in one
torch.save file loaded fully into memory -- there is nothing to gain
from a columnar/out-of-core format at this size. If this is later scaled
up by another order of magnitude or two (many more collocation points or
configurations, to the point where loading the whole file into RAM stops
being practical), Parquet is the better choice -- true column-chunked,
partial reads without loading everything at once -- at the cost of a new
dependency this project doesn't currently have.

Schema (one dict, torch.save'd):
  "configuration_features" : (num_configs, num_res, 5) float32, [m,k,f_t,x,y]
  "frequency_values"       : (num_freqs,) float32, Hz
  "collocation_points"     : (num_configs, points_per_config, 2) float32, [x,y] meters
  "real_displacement"      : (num_configs, points_per_config, num_freqs) float32
  "imag_displacement"      : (num_configs, points_per_config, num_freqs) float32
  "num_res", "points_per_config", "seed", "Lx", "Ly", "feature_layout": metadata

Collocation points are resampled independently per configuration (not a
shared fixed grid) -- matching the random-scatter collocation strategy
in the reference paper this project's physics-informed discussion has
been referring to (Dogu et al., INTER-NOISE 2026, slide 12), rather than
a dense regular mesh, which would blow up storage for no real benefit
(the displacement field is smooth in space away from resonator
attachment points; a modest random scatter captures it as well as a
dense grid would for training purposes).
"""

from __future__ import annotations

import time

import numpy as np

from utils.erp_dataset import configuration_to_resonators
from utils.physics import Lx, Ly, edge_margin
from utils.physics import freqs as default_freqs
from utils.physics import num_res as default_num_res
from utils.physics import resonator_bounds
from utils.solver import compute_displacement
from utils.support import lhs_sampling, save_dataset


def generate_field_dataset(
    num_configurations: int,
    points_per_config: int = 150,
    num_res: int = default_num_res,
    frequencies: np.ndarray | None = None,
    seed: int = 727,
    filename: str | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    """Generate the (configuration, frequency, position) -> (Re w, Im w) dataset.

    ``points_per_config`` collocation points are drawn uniformly at random
    over the plate interior (same edge margin used for resonator
    placement) independently for each configuration. Compute cost is
    dominated by the per-frequency linear solve inside
    ``compute_displacement`` (one per configuration x frequency,
    independent of how many spatial points are then evaluated), so this
    is only marginally more expensive than generating an ERP-only
    dataset of the same configuration count.
    """
    frequencies = np.asarray(default_freqs if frequencies is None else frequencies, dtype=np.float32)
    n_freqs = frequencies.size
    num_configurations = int(num_configurations)
    points_per_config = int(points_per_config)

    design_samples = lhs_sampling(num_configurations, bounds=resonator_bounds(num_res), seed=seed)

    configuration_features = np.empty((num_configurations, num_res, 5), dtype=np.float32)
    collocation_points = np.empty((num_configurations, points_per_config, 2), dtype=np.float32)
    real_displacement = np.empty((num_configurations, points_per_config, n_freqs), dtype=np.float32)
    imag_displacement = np.empty((num_configurations, points_per_config, n_freqs), dtype=np.float32)

    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    log_every = max(1, num_configurations // 20)
    for i in range(num_configurations):
        quad = design_samples[i].reshape(num_res, 4)
        x_r, y_r, f_t, m = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
        k = m * (2.0 * np.pi * f_t) ** 2
        features = np.stack([m, k, f_t, x_r, y_r], axis=1).astype(np.float32)
        configuration_features[i] = features
        resonators = configuration_to_resonators(features)

        px = rng.uniform(edge_margin, Lx - edge_margin, size=points_per_config)
        py = rng.uniform(edge_margin, Ly - edge_margin, size=points_per_config)
        collocation_points[i, :, 0] = px
        collocation_points[i, :, 1] = py

        w = compute_displacement(resonators, x=px, y=py, frequencies=frequencies)  # (points_per_config, n_freqs), complex
        real_displacement[i] = w.real.astype(np.float32)
        imag_displacement[i] = w.imag.astype(np.float32)

        if verbose and ((i + 1) % log_every == 0 or i + 1 == num_configurations):
            elapsed = time.perf_counter() - start
            print(f"Configuration {i + 1}/{num_configurations} complete (elapsed {elapsed:.2f} s)")

    dataset: dict[str, object] = {
        "configuration_features": configuration_features,
        "frequency_values": frequencies,
        "collocation_points": collocation_points,
        "real_displacement": real_displacement,
        "imag_displacement": imag_displacement,
        "num_res": int(num_res),
        "points_per_config": points_per_config,
        "seed": int(seed),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "feature_layout": "per_resonator_[m,k,f_t,x,y]",
    }
    if filename is not None:
        save_dataset(dataset, filename)
    return dataset


def load_field_dataset(filename: str) -> dict[str, object]:
    from utils.support import load_dataset

    return load_dataset(filename)
