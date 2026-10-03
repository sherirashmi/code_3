"""Run the trained position-only models on hand-picked configurations that are
not in the dataset.

For each configuration (resonator positions; m and f_t are the dataset's fixed
values): the true ERP is computed with the solver, the models see its 40-120 Hz
band, sample position sets, and every sample is checked with the solver again.

Saves to ``plots/<dataset>/custom_configurations/``:

* ``custom_config_<i>.png`` -- per configuration: true vs predicted ERP (the
  primary model's own pick and best of N) and the plate with the true positions,
  their y-mirror images, every sample and the picks of all models;
* ``custom_configurations.png`` -- all configurations in one figure;
* ``custom_configurations.txt`` -- the numbers for every model.

Usage (from the repository root)::

    python -m 2_res_erp_inverse_models.predict_custom
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_inverse_operators.evaluate import SPAWN_CONTEXT, solve_configs
from utils.erp_dataset import DATASETS, normalize_erp_array, select_dataset_modal_resolution
from utils.physics import Ly
from utils.plotting import ERP_LABEL, FREQ_LABEL, save_figure
from utils.support import device, load_dataset

from .common import DEFAULT_DATASET, band_mask, model_path, plot_dir
from .design_space import sort_by_x
from .evaluate import RES_COLORS, _draw_plate, load_model, sample_designs

# (title, [(x1, y1), (x2, y2)]) in metres. None of these is in the dataset.
CUSTOM_CONFIGURATIONS = [
    ("Ordinary positions, not in the dataset", [(0.30, 0.12), (1.10, 0.38)]),
    ("Two resonators almost on top of each other", [(0.70, 0.20), (0.74, 0.22)]),
    ("One resonator next to the excitation force", [(0.88, 0.31), (0.20, 0.42)]),
    ("Near the corners, outside the sampled region", [(0.02, 0.03), (1.38, 0.47)]),
]
MODELS = ("Flow", "MDN", "Diffusion")  # first = the model whose ERP is drawn
MODEL_MARKERS = {"Flow": "X", "MDN": "P", "Diffusion": "D"}
NUM_SAMPLES = 32


def build_design(positions, m: float, f_t: float) -> np.ndarray:
    rows = [[m, m * (2 * np.pi * f_t) ** 2, f_t, x, y] for x, y in positions]
    return sort_by_x(np.asarray(rows, dtype=np.float64))


def main(dataset_tag: str = DEFAULT_DATASET, configurations=CUSTOM_CONFIGURATIONS, models=MODELS,
         num_samples: int = NUM_SAMPLES) -> None:
    select_dataset_modal_resolution(dataset_tag)
    models = [m for m in models if model_path(m, dataset_tag).exists()]
    loaded = {m: load_model(m, dataset_tag) for m in models}
    norm = loaded[models[0]][1]
    freq = np.asarray(load_dataset(DATASETS[dataset_tag]["files"][0])["frequency_values"], dtype=np.float64)
    mask = band_mask(freq, float(norm["f_low"]), float(norm["f_high"]))
    f_low, f_high = float(norm["f_low"]), float(norm["f_high"])
    designs = np.stack([build_design(p, float(norm["fixed_m"]), float(norm["fixed_f_t"])) for _, p in configurations])
    n, num_res = designs.shape[0], designs.shape[1]
    out_dir = plot_dir(dataset_tag, "custom_configurations")

    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT) as pool:
        true_erp = solve_configs(pool, designs, freq)  # (n, n_freq) dB
        spectrum = torch.from_numpy(normalize_erp_array(true_erp, norm)[:, mask]).float().to(device)
        results = {}
        for name, (model, model_norm) in loaded.items():
            samples, log_p = sample_designs(model, spectrum, num_samples, model_norm)
            solved = solve_configs(pool, samples.reshape(n * num_samples, num_res, 5), freq).reshape(n, num_samples, -1)
            band_mse = ((solved[..., mask] - true_erp[:, None, mask]) ** 2).mean(-1)
            own = log_p.argmax(-1) if log_p is not None else np.zeros(n, dtype=int)
            results[name] = {"samples": samples, "solved": solved, "own": own, "best": band_mse.argmin(-1),
                             "own_label": "own pick (max. log p)" if log_p is not None else "first sample"}

    def distance_cm(pred, true):
        return 100 * np.hypot(pred[:, 3] - true[:, 3], pred[:, 4] - true[:, 4])

    def mirrored(true):
        out = true.copy()
        out[:, 4] = Ly - out[:, 4]
        return out

    lines = [f"Position-only inverse models on hand-picked configurations ({num_samples} samples each, solver-checked)",
             f"Fixed resonators: m = {norm['fixed_m']:g} kg, f_t = {norm['fixed_f_t']:g} Hz; model input: ERP {f_low:g}-{f_high:g} Hz", ""]
    primary = models[0]
    for i, (title, _) in enumerate(configurations):
        true = designs[i]
        band_rmse = lambda s, r: np.sqrt(((r["solved"][i, s] - true_erp[i])[mask] ** 2).mean())
        lines.append(f"Configuration {i + 1}: {title}")
        lines.append("  true: " + "; ".join(f"R{r + 1} ({true[r, 3]:.2f}, {true[r, 4]:.2f}) m" for r in range(num_res)))
        for name, res in results.items():
            for rule, key in (("own", "own"), ("best of N", "best")):
                pick = res["samples"][i, res[key][i]]
                d, dm = distance_cm(pick, true), distance_cm(pick, mirrored(true))
                lines.append(f"  {name:<9} {rule:<9}: " + "; ".join(f"R{r + 1} ({pick[r, 3]:.2f}, {pick[r, 4]:.2f}) m" for r in range(num_res))
                             + f" | error {', '.join(f'{v:.1f}' for v in d)} cm (to y-mirror: {', '.join(f'{v:.1f}' for v in dm)} cm)"
                             + f" | band ERP RMSE {band_rmse(res[key][i], res):.2f} dB")
        lines.append("")

        res = results[primary]
        fig, (ax, ax_plate) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.0, 1.05]})
        ax.axvspan(f_low, f_high, color="#e8eef8", zorder=0, label=f"ERP band seen by the model ({f_low:g}--{f_high:g} Hz)")
        for s in range(num_samples):
            ax.plot(freq, res["solved"][i, s], color="0.8", lw=0.6, zorder=1, label=f"All {num_samples} {primary} samples" if s == 0 else None)
        ax.plot(freq, true_erp[i], color="black", lw=2.0, zorder=4, label="True ERP of the new configuration")
        o, b = res["own"][i], res["best"][i]
        ax.plot(freq, res["solved"][i, o], color="#2a78d6", lw=1.6, zorder=3,
                label=f"{primary}, {res['own_label']} (band RMSE {band_rmse(o, res):.2f} dB)")
        ax.plot(freq, res["solved"][i, b], color="#eb6834", lw=1.4, ls="--", zorder=3,
                label=f"{primary}, best of {num_samples} (band RMSE {band_rmse(b, res):.2f} dB)")
        ax.set_xlim(freq.min(), freq.max())
        ax.set_xlabel(FREQ_LABEL)
        ax.set_ylabel(ERP_LABEL)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8.5)
        ax.set_title("ERP")

        _draw_plate(ax_plate, true, samples=res["samples"][i])
        mirror = mirrored(true)
        for r in range(num_res):
            ax_plate.plot(mirror[r, 3], mirror[r, 4], "o", mfc="none", mec=RES_COLORS[r], ms=11, mew=1.3, ls="",
                          zorder=4, label=f"Mirror image of true resonator {r + 1} ($y \\to L_y - y$)")
        for name, other in results.items():
            pick = other["samples"][i, other["own"][i]]
            for r in range(num_res):
                ax_plate.plot(pick[r, 3], pick[r, 4], MODEL_MARKERS[name], color="white", mec=RES_COLORS[r], mew=1.8,
                              ms=9, ls="", zorder=7, label=f"{name} {other['own_label'].split(' (')[0]}" if r == 0 else None)
        handles, labels = ax_plate.get_legend_handles_labels()
        keep = {}
        for h, l in zip(handles, labels):
            keep.setdefault(l, h)
        ax_plate.legend(keep.values(), keep.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
                        frameon=False, fontsize=8)
        ax_plate.set_title("Resonator positions (true: filled circles; model picks: white markers)")
        fig.suptitle(f"New configuration {i + 1}: {title}", fontsize=13)
        fig.tight_layout()
        save_figure(fig, out_dir / f"custom_config_{i + 1}.png")

    fig, axes = plt.subplots(n, 2, figsize=(15, 4.3 * n), gridspec_kw={"width_ratios": [1.0, 1.05]}, squeeze=False)
    for i, (title, _) in enumerate(configurations):
        res = results[primary]
        ax, ax_plate = axes[i]
        ax.axvspan(f_low, f_high, color="#e8eef8", zorder=0)
        ax.plot(freq, true_erp[i], color="black", lw=1.8, label="True ERP")
        ax.plot(freq, res["solved"][i, res["own"][i]], color="#2a78d6", lw=1.4, label=f"{primary}, {res['own_label']}")
        ax.plot(freq, res["solved"][i, res["best"][i]], color="#eb6834", lw=1.2, ls="--", label=f"{primary}, best of {num_samples}")
        ax.set_ylabel(ERP_LABEL)
        ax.set_title(f"{i + 1}. {title}", fontsize=10)
        ax.grid(alpha=0.3)
        _draw_plate(ax_plate, designs[i], samples=res["samples"][i], picked=res["samples"][i, res["own"][i]], picked_label=primary)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[-1, 0].set_xlabel(FREQ_LABEL)
    fig.suptitle(f"{primary} on new configurations: predicted vs true ERP and resonator positions", fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir / "custom_configurations.png")
    (out_dir / "custom_configurations.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
