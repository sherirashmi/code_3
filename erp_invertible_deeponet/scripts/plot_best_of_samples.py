"""Target ERP and the best of the sampled designs of every variant, in one plot.

For the first ``num_examples`` test targets (the same targets as evaluate.py's
inverse examples), every variant draws ``num_samples`` designs (fixed seed,
so the figure is reproducible), the solver computes their ERPs, and the one
closest to the target (lowest RMSE) is drawn -- one panel per target, one
line per variant.

Saves plots/<dataset>/ALL_MODELS/best_of_samples_all_variants.png.

Usage:  python -m erp_invertible_deeponet.scripts.plot_best_of_samples [Q8 Q64 Q128 ...]
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

from erp_inverse.scripts.common import denormalize_design, sort_resonators_by_ft
from erp_inverse.scripts.evaluate import SPAWN_CONTEXT, solve_configs
from utils.erp_dataset import denormalize_erp_array
from utils.plotting import ERP_LABEL, FREQ_LABEL, save_figure
from utils.support import device

from .evaluate import COLORS
from .train import DATASET, VARIANTS, load_variant, model_path, plot_dir, prepare

DEFAULT = ("Q8", "Q64", "Q128", "Q64-FNO", "Q64-DCO", "Q64-DNO", "Q64-WNO", "Q64-LNO", "Q64-SIREN")


def main(names, num_examples: int = 3, num_samples: int = 16, seed: int = 727, dataset_tag: str = DATASET) -> None:
    names = [n for n in names if n in VARIANTS and model_path(n, dataset_tag).exists()]
    dataset, loaders = prepare(dataset_tag)
    freq = np.asarray(dataset.frequency_values, dtype=np.float64)
    num_res = int(dataset.num_res)
    spectra = torch.cat([s for s, _ in loaders["test"]])[:num_examples]
    best, rmse = {}, {}
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT) as pool:
        for name in names:
            model, norm = load_variant(name, dataset_tag)
            torch.manual_seed(seed)
            with torch.no_grad():
                flat = model.sample(spectra.to(device), num_samples).cpu().numpy()
            physical = denormalize_design(flat, num_res, norm)
            physical = sort_resonators_by_ft(physical.reshape(-1, num_res, 5))
            solved = solve_configs(pool, physical, freq).reshape(num_examples, num_samples, -1)
            target = denormalize_erp_array(spectra.numpy(), norm).astype(np.float64)
            err = np.sqrt(((solved - target[:, None]) ** 2).mean(-1))
            pick = err.argmin(1)
            best[name] = solved[np.arange(num_examples), pick]
            rmse[name] = err[np.arange(num_examples), pick]
            print(f"{name}: best-of-{num_samples} RMSE " + ", ".join(f"{v:.2f}" for v in rmse[name]) + " dB", flush=True)

    fig, axes = plt.subplots(num_examples, 1, figsize=(13, 4.2 * num_examples), sharex=True)
    for k, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(freq, target[k], color="black", lw=2.6, label="Target ERP", zorder=5)
        for name in names:
            ax.plot(freq, best[name][k], color=COLORS.get(name, "#6f6f6a"), lw=1.2,
                    ls="--" if name == "Q8" else "-", label=f"{name} ({rmse[name][k]:.2f} dB)")
        ax.set_ylabel(ERP_LABEL)
        ax.set_title(f"Test configuration {k + 1}")
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False, title=f"Best of {num_samples} (RMSE)",
                  title_fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel(FREQ_LABEL)
    fig.tight_layout()
    path = plot_dir(dataset_tag) / "best_of_samples_all_variants.png"
    save_figure(fig, path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = DATASET
    if "--dataset" in args:
        i = args.index("--dataset")
        tag = args[i + 1]
        args = args[:i] + args[i + 2:]
    main(args or DEFAULT, dataset_tag=tag)
