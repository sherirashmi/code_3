"""Predicted vs. true ERP spectra for a few held-out configurations.

For each of the first ``num_configs`` test targets of the model's dataset:
sample ``num_samples`` designs, run them through the actual solver, and plot
the target ERP against

* the model's own target-blind pick (highest log p for density models,
  the first sample for Diffusion / PadINN), and
* the best of the samples by solver error (uses the target: an upper bound),

with the true and both predicted designs listed under each panel. Saves
``plots/<dataset>/<MODEL>/erp_spectrum_test_config_0<i>.png`` (one per
target) and ``erp_spectrum_test_configs.png`` (all targets in one figure).

Usage: ``python -m erp_inverse.scripts.spectrum_examples <MODEL> <dataset tag>``
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_inverse.scripts.evaluate import (
    SPAWN_CONTEXT,
    _check_norm,
    load_inverse_model,
    score_samples,
    selection_indices,
    solve_configs,
)
from erp_inverse.scripts.registry import parse_variant
from utils.erp_dataset import DATASETS, denormalize_erp_array, select_dataset_modal_resolution
from utils.paths import inverse_plot_dir
from utils.plotting import ERP_LABEL, FREQ_LABEL, save_figure
from utils.support import device


def _design_text(label: str, design: np.ndarray) -> str:
    parts = [
        f"R{i + 1}: m={r[0]:.2f} kg, $f_t$={r[2]:.1f} Hz, (x, y)=({r[3]:.2f}, {r[4]:.2f}) m"
        for i, r in enumerate(design)
    ]
    return f"{label}: " + "; ".join(parts)


def main(model_name: str, dataset_tag: str, num_configs: int = 3, num_samples: int = 16) -> list[str]:
    select_dataset_modal_resolution(dataset_tag)
    files = DATASETS[dataset_tag]["files"]
    dataset, loaders = prepare_inverse_data(
        num_configurations=int(DATASETS[dataset_tag]["num_configurations"]), batch_size=64,
        dataset_file=files if len(files) > 1 else files[0], seed=727,
        design_param=parse_variant(model_name)[2],
    )
    norm = dataset.norm_params
    num_res = int(norm["num_res"])
    freq_hz = np.asarray(dataset.frequency_values)
    spectrum, design = next(iter(loaders["test"]))
    spectrum, design = spectrum[:num_configs], design[:num_configs]
    target_db = denormalize_erp_array(spectrum.numpy(), norm)
    true_designs = denormalize_design(design.reshape(num_configs, -1).numpy(), num_res, norm, consistent=False)

    model, model_norm = load_inverse_model(model_name, dataset_tag)
    _check_norm(model_name, model_norm, norm)
    spectrum_dev = spectrum.to(device)
    with torch.no_grad():
        result = model.sample(spectrum_dev, num_samples=num_samples)
    flat = (result[0] if isinstance(result, tuple) else result).to(device)
    physical, log_p = score_samples(model, spectrum_dev, flat, norm)  # (n, S, R, 5)

    pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT)
    try:
        solved = solve_configs(pool, physical.reshape(num_configs * num_samples, num_res, 5), freq_hz)
    finally:
        pool.shutdown()
    solved = solved.reshape(num_configs, num_samples, -1)
    mse = ((solved - target_db[:, None, :]) ** 2).mean(axis=-1)
    picks = selection_indices(model_name, log_p, mse)
    own = picks["own"] if picks["own"] is not None else picks["random"]
    own_label = "own pick (max log p)" if picks["own"] is not None else "random sample"
    best = picks["oracle"]

    out_dir = inverse_plot_dir(dataset_tag, model_name)
    saved = []

    def draw(ax, i):
        rows = np.arange(num_samples)
        for s in rows:
            ax.plot(freq_hz, solved[i, s], color="0.75", lw=0.7, zorder=1)
        ax.plot(freq_hz, target_db[i], color="black", lw=2.0, label="True ERP (target)", zorder=4)
        ax.plot(freq_hz, solved[i, own[i]], color="#2a78d6", lw=1.6,
                label=f"Predicted, {own_label}: RMSE {np.sqrt(mse[i, own[i]]):.2f} dB", zorder=3)
        ax.plot(freq_hz, solved[i, best[i]], color="#eb6834", lw=1.4, ls="--",
                label=f"Predicted, best of {num_samples}: RMSE {np.sqrt(mse[i, best[i]]):.2f} dB", zorder=3)
        ax.set_xlabel(FREQ_LABEL)
        ax.set_ylabel(ERP_LABEL)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        text = "\n".join([
            _design_text("True", true_designs[i]),
            _design_text("Own pick", sort_resonators_by_ft(physical[i, own[i]][None])[0]),
            _design_text("Best", sort_resonators_by_ft(physical[i, best[i]][None])[0]),
        ])
        ax.text(0.0, -0.22, text, transform=ax.transAxes, fontsize=7, va="top", family="monospace")

    for i in range(num_configs):
        fig, ax = plt.subplots(figsize=(9, 6.2))
        draw(ax, i)
        ax.set_title(f"{model_name}: predicted vs true ERP, test configuration {i + 1} "
                     f"(grey: all {num_samples} samples)")
        fig.tight_layout()
        path = out_dir / f"erp_spectrum_test_config_{i + 1:02d}.png"
        save_figure(fig, path)
        plt.close(fig)
        saved.append(str(path))

    fig, axes = plt.subplots(num_configs, 1, figsize=(9, 6.0 * num_configs))
    for i, ax in enumerate(np.atleast_1d(axes)):
        draw(ax, i)
        ax.set_title(f"Test configuration {i + 1}")
    fig.suptitle(f"{model_name}: predicted vs true ERP spectra ({dataset_tag})")
    fig.tight_layout()
    path = out_dir / "erp_spectrum_test_configs.png"
    save_figure(fig, path)
    plt.close(fig)
    saved.append(str(path))
    for i in range(num_configs):
        print(f"{model_name} config {i + 1}: own-pick RMSE {np.sqrt(mse[i, own[i]]):.2f} dB, "
              f"best-of-{num_samples} RMSE {np.sqrt(mse[i, best[i]]):.2f} dB")
    return saved


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
