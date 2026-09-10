"""Aggregate prediction-quality stats for the 4 trained inverse models.

Mirrors the forward operators' own evaluation convention (RMSE/MAE in dB,
Pearson r, R^2, prediction-vs-ground-truth scatter) so the two are directly
comparable in form. The inverse problem has no single "ground truth design"
to regress against (see module docstrings elsewhere in this package), so the
comparison is done the only way that is actually meaningful here: sample a
design from each model for a held-out target spectrum, run it through the
*actual coupled plate-resonator solver* (``utils.solver.compute_erp_spectrum``
-- the same modal solver that generated every ERP value in the dataset, not
a neural surrogate), and compare the resulting spectrum to the true target
spectrum. This removes a forward operator's own approximation error from
the inverse-model evaluation entirely -- the reported numbers reflect only
how good the sampled *design* is, not how good some other network's guess
at its spectrum is.

The solver is exact but far slower than a forward-operator inference pass
(~0.25s per configuration on this CPU-only environment, vs. instant for a
neural net), so solver calls are parallelized across all CPU cores and the
number of test examples/samples is kept modest to finish in a reasonable
time.

Design selection per target (matches inverse_operators/train_all.py):
  - MDN / cVAE / Flow have a tractable log p(design | spectrum); the
    reported prediction is the sample with the highest log-probability
    under the model itself (does not look at the ground truth spectrum).
  - Diffusion has no tractable density; its reported prediction is the
    sample with the lowest solver-simulated reconstruction error against
    the target (this one *does* use the target for selection, since it is
    the only score available -- flagged here as it was in train_all.py, not
    swept under the rug).
"""

from __future__ import annotations

import multiprocessing
import os

# Must be set before numpy/scipy are imported by worker processes -- each
# solver call is itself a sequence of BLAS-backed linalg.solve calls, and
# BLAS defaults to using every core it can see. Four worker processes each
# trying to multithread across all 4 cores thrashes badly (measured: the
# "parallel" run was *slower* than serial); pinning each worker to one BLAS
# thread is what actually lets the process pool use the 4 cores cleanly.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# The default 'fork' start method clones this already-running process,
# including whatever BLAS thread pool numpy already initialized *before*
# this module got a chance to set the env vars above -- fork inherits that
# stale thread count. 'spawn' starts each worker as a genuinely fresh
# interpreter that imports numpy from scratch and picks up the env vars
# for real; it's what makes the single-thread pin above actually apply.
SPAWN_CONTEXT = multiprocessing.get_context("spawn")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from inverse_operators.common import denormalize_design, prepare_inverse_data
from inverse_operators.cvae import ConditionalVAE
from inverse_operators.diffusion import ConditionalDiffusion
from inverse_operators.flow import ConditionalFlow
from inverse_operators.mdn import MDN
from utils.erp_dataset import configuration_to_resonators, denormalize_erp_array
from utils.solver import compute_erp_spectrum

NUM_RES = 3
DESIGN_DIM = NUM_RES * 5
OUT_DIR = Path("plots/INVERSE_OPERATORS")

MODEL_BUILDERS = {
    "MDN": lambda: MDN(design_dim=DESIGN_DIM, num_components=10),
    "cVAE": lambda: ConditionalVAE(design_dim=DESIGN_DIM, latent_dim=8),
    "Flow": lambda: ConditionalFlow(design_dim=DESIGN_DIM, num_layers=8, hidden=96),
    "Diffusion": lambda: ConditionalDiffusion(design_dim=DESIGN_DIM, num_steps=100, hidden=128),
}


def load_inverse_model(name: str):
    checkpoint = torch.load(f"models/inverse_{name.lower()}.pth", map_location="cpu", weights_only=False)
    model = MODEL_BUILDERS[name]()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["norm_params"]


def pick_best_indices(has_log_prob: bool, log_probs: np.ndarray | None, recon_mse: np.ndarray) -> np.ndarray:
    """Per-example index of the reported prediction among the sampled candidates.

    ``log_probs``/``recon_mse``: (b, num_samples). Returns (b,) int array.
    """
    if has_log_prob:
        return log_probs.argmax(axis=-1)
    return recon_mse.argmin(axis=-1)


def _solve_one(args: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Worker: one physical (num_res,5) config -> real ERP spectrum in dB."""
    config_physical, frequency_values = args
    resonators = configuration_to_resonators(config_physical)
    return compute_erp_spectrum(resonators, frequencies=frequency_values)


def solve_configs(pool: ProcessPoolExecutor, configs_physical: np.ndarray, frequency_values: np.ndarray) -> np.ndarray:
    """``(n, num_res, 5)`` physical configs -> ``(n, n_freq)`` real ERP spectra (dB), via the actual solver."""
    jobs = [(configs_physical[i], frequency_values) for i in range(configs_physical.shape[0])]
    results = list(pool.map(_solve_one, jobs, chunksize=4))
    return np.stack(results, axis=0)


def main(
    num_test_examples: int = 150,
    num_samples: int = 8,
    num_configurations: int = 100000,
    dataset_file=(
        "datasets/dataset_erp_ft_100k_part1.pth",
        "datasets/dataset_erp_ft_100k_part2.pth",
    ),
):
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=64,
        dataset_file=list(dataset_file), seed=727,
    )
    norm = dataset.norm_params
    freq_hz = np.asarray(dataset.frequency_values)

    test_spectrum = []
    for spectrum, _design in loaders["test"]:
        test_spectrum.append(spectrum)
        if sum(s.shape[0] for s in test_spectrum) >= num_test_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:num_test_examples]
    n = test_spectrum.shape[0]
    true_erp_db = denormalize_erp_array(test_spectrum.numpy(), norm)  # (n, n_freq)

    stats = {}
    predictions_db = {}
    num_workers = max(1, os.cpu_count() or 1)
    print(f"Using {num_workers} worker processes for the physics solver.")

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=SPAWN_CONTEXT) as pool:
        for name in MODEL_BUILDERS:
            print(f"Evaluating {name} ({n} targets x {num_samples} samples, actual solver) ...")
            model, _ = load_inverse_model(name)

            with torch.no_grad():
                result = model.sample(test_spectrum, num_samples=num_samples)
            has_log_prob = isinstance(result, tuple)
            flat_samples = result[0] if has_log_prob else result  # (n, num_samples, design_dim)
            log_probs = result[1].numpy() if has_log_prob else None

            physical = denormalize_design(flat_samples.numpy(), NUM_RES, norm)  # (n, num_samples, num_res, 5)
            flat_physical = physical.reshape(n * num_samples, NUM_RES, 5)
            solved = solve_configs(pool, flat_physical, freq_hz)  # (n*num_samples, n_freq)
            solved = solved.reshape(n, num_samples, -1)

            target = true_erp_db[:, None, :]
            recon_mse = ((solved - target) ** 2).mean(axis=-1)  # (n, num_samples), real dB^2

            best_idx = pick_best_indices(has_log_prob, log_probs, recon_mse)  # (n,)
            best_pred = solved[np.arange(n), best_idx]  # (n, n_freq)
            predictions_db[name] = best_pred

            error = best_pred - true_erp_db
            mae = np.abs(error).mean()
            rmse = np.sqrt((error ** 2).mean())
            r_global = np.corrcoef(best_pred.ravel(), true_erp_db.ravel())[0, 1]
            per_example_r = np.array([
                np.corrcoef(best_pred[i], true_erp_db[i])[0, 1] for i in range(n)
            ])
            ss_res = (error ** 2).sum()
            ss_tot = ((true_erp_db - true_erp_db.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot

            stats[name] = {
                "mae_db": float(mae),
                "rmse_db": float(rmse),
                "pearson_r": float(r_global),
                "mean_spectrum_r": float(np.nanmean(per_example_r)),
                "r2": float(r2),
            }
            print(f"  {name}: MAE={mae:.3f} dB  RMSE={rmse:.3f} dB  Pearson r={r_global:.4f}  R^2={r2:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- summary table ----
    lines = [
        f"Inverse-design prediction quality ({n} held-out target spectra, "
        f"best of {num_samples} samples per target, scored with the actual "
        f"coupled plate-resonator solver -- not a neural forward surrogate)",
        "=" * 100,
        f"{'Model':<10} {'MAE (dB)':>10} {'RMSE (dB)':>10} {'Pearson r':>10} {'Mean spec. r':>13} {'R^2':>8}",
        "-" * 100,
    ]
    for name, s in stats.items():
        lines.append(
            f"{name:<10} {s['mae_db']:>10.3f} {s['rmse_db']:>10.3f} {s['pearson_r']:>10.4f} "
            f"{s['mean_spectrum_r']:>13.4f} {s['r2']:>8.4f}"
        )
    table_text = "\n".join(lines)
    (OUT_DIR / "prediction_stats.txt").write_text(table_text + "\n")
    print("\n" + table_text)

    # ---- bar charts: MAE, Pearson r, R^2 per method ----
    names = list(stats.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    axes[0].bar(names, [stats[m]["mae_db"] for m in names], color=colors)
    axes[0].set_ylabel("MAE (dB)")
    axes[0].set_title("Prediction MAE (lower better)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(names, [stats[m]["pearson_r"] for m in names], color=colors)
    axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Global Pearson correlation (higher better)")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(names, [stats[m]["r2"] for m in names], color=colors)
    axes[2].set_ylabel("R^2")
    axes[2].set_title("Coefficient of determination (higher better)")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Inverse-design methods: solver-scored prediction quality on {n} held-out spectra",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "prediction_stats_bars.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'prediction_stats_bars.png'}")

    # ---- prediction vs ground truth scatter, one panel per method ----
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharex=True, sharey=True)
    lo = min(true_erp_db.min(), *(predictions_db[m].min() for m in names))
    hi = max(true_erp_db.max(), *(predictions_db[m].max() for m in names))
    rng = np.random.default_rng(0)
    for ax, name, color in zip(axes, names, colors):
        true_flat = true_erp_db.ravel()
        pred_flat = predictions_db[name].ravel()
        if true_flat.size > 6000:
            pick = rng.choice(true_flat.size, size=6000, replace=False)
            true_flat, pred_flat = true_flat[pick], pred_flat[pick]
        ax.scatter(true_flat, pred_flat, s=4, alpha=0.25, color=color, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
        ax.set_title(f"{name}\nMAE={stats[name]['mae_db']:.2f} dB, r={stats[name]['pearson_r']:.3f}")
        ax.set_xlabel("Ground truth ERP (dB)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Predicted ERP (dB)\n(actual solver, not a neural surrogate)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Inverse-design prediction vs. ground truth ERP (solver-scored)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_DIR / "prediction_vs_ground_truth.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'prediction_vs_ground_truth.png'}")

    return stats


if __name__ == "__main__":
    main()
