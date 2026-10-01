"""Regenerate the combined 7-model ERP-SPECTRUM (frequency vs. dB curve,
not a scatter of points) validation figure, using each model's CURRENTLY
saved checkpoint (loaded, not retrained) -- the committed version predates
last night's 6-model retrain and today's SurrogateInverse rework entirely.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/home/user/code_3")

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_inverse_operators.common import denormalize_design, prepare_inverse_data
from erp_inverse_operators.evaluate import MODEL_BUILDERS, SPAWN_CONTEXT, load_inverse_model, solve_configs
from erp_inverse_operators.registry import NUM_RES
from utils.erp_dataset import denormalize_erp_array
from utils.support import device

OUT_DIR = Path("erp_inverse_operators/plots/ALL_MODELS")
NUM_EXAMPLES = 3
NUM_SAMPLES = 6


def main():
    dataset, loaders = prepare_inverse_data(
        num_configurations=100000, batch_size=64,
        dataset_file=[
            "datasets/dataset_erp_ft_100k_part1.pth",
            "datasets/dataset_erp_ft_100k_part2.pth",
        ],
        seed=727,
    )
    norm = dataset.norm_params
    freq_hz = np.asarray(dataset.frequency_values)

    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= NUM_EXAMPLES:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:NUM_EXAMPLES]
    true_erp = denormalize_erp_array(test_spectrum.numpy(), norm)
    test_spectrum = test_spectrum.to(device)

    names = list(MODEL_BUILDERS)
    models = {}
    for name in names:
        print(f"Loading {name} ...")
        model, _ = load_inverse_model(name)
        models[name] = model

    solver_pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT)

    fig, axes = plt.subplots(NUM_EXAMPLES, len(names), figsize=(4.8 * len(names), 4.2 * NUM_EXAMPLES))
    for row in range(NUM_EXAMPLES):
        for col, name in enumerate(names):
            model = models[name]
            model.eval()
            with torch.no_grad():
                result = model.sample(test_spectrum[row : row + 1], num_samples=NUM_SAMPLES)
            has_log_prob = isinstance(result, tuple)
            flat_samples = (result[0][0] if has_log_prob else result[0]).cpu()
            log_probs = (result[1][0].cpu() if has_log_prob else None)

            physical = denormalize_design(flat_samples.numpy(), NUM_RES, norm)
            predicted_erp = solve_configs(solver_pool, physical, freq_hz)
            recon_mse = ((predicted_erp - true_erp[row][None, :]) ** 2).mean(axis=1)
            best_idx = int(log_probs.numpy().argmax()) if has_log_prob else int(recon_mse.argmin())

            ax = axes[row, col] if NUM_EXAMPLES > 1 else axes[col]
            for i in range(NUM_SAMPLES):
                if i == best_idx:
                    continue
                ax.plot(freq_hz, predicted_erp[i], color="#4C72B0", alpha=0.3, lw=1.1)
            ax.plot(freq_hz, predicted_erp[best_idx], color="#C44E52", lw=2.0,
                     label="Best sample" if row == 0 and col == 0 else None)
            ax.plot(freq_hz, true_erp[row], color="black", lw=2,
                     label="Target" if row == 0 and col == 0 else None)
            ax.set_title(f"{name}\nMSE={recon_mse[best_idx]:.3g}" if row == 0 else f"MSE={recon_mse[best_idx]:.3g}",
                         fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Example {row + 1}\nERP (dB)")
            if row == NUM_EXAMPLES - 1:
                ax.set_xlabel("Frequency (Hz)")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"Inverse-design validation: {NUM_SAMPLES} sampled designs per method "
        "(best-scoring highlighted red), run through the actual solver (not a neural surrogate)\n"
        "Using each model's currently-saved checkpoint",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "validation_reconstructions.png", dpi=150)
    plt.close(fig)
    solver_pool.shutdown()
    print(f"Saved {OUT_DIR / 'validation_reconstructions.png'}")


if __name__ == "__main__":
    main()
