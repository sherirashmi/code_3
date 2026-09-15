"""Inverse-model diagnostics matching Dogu et al., "Invertible Neural
Operators for Generative Design of Acoustic Metamaterials" (INTER-NOISE
2026) -- the same TVA-mass / ERP-spectrum problem, DeepONet-branch-
conditioned invertible flow architecture (i.e. the Kaltenbach et al. line
this project's BasisFlow/PadINN already draw on), reporting the design's
own metrics/plots rather than this project's own. This module borrows
their evaluation *methodology* only, not their numbers or their model.

Four pieces, each a direct analogue of one of their reported results:

  1. Posterior calibration curve (their Fig. 5): for a swept nominal
     coverage level p, what fraction of true design values actually fall
     inside the model's own p% credible interval? A perfectly calibrated
     model traces the diagonal; a curve above it (like theirs) means the
     model is conservative (uncertainty is never overconfident); below
     means overconfident. Computed per design dimension marginally (their
     2-mass problem could afford a joint 2D ellipse; this project's
     15-dim design can't reliably estimate a joint credible region from a
     few hundred samples, so this uses per-dimension percentile
     intervals -- the standard practical compromise).

  2. Stratified recovery table (their Table 1): the same MAE this
     project already reports in design_parameter_stats.txt, but broken
     out by physical difficulty regime instead of one pooled number --
     well-separated vs. near-degenerate resonator tuning (the exact
     degeneracy this project's own docstrings keep citing as the reason
     the inverse problem is non-unique), and heavy vs. light mass.
     Thresholds are this project's own (see _difficulty_groups), not
     theirs -- their band/groups don't transfer directly (see below).

  3. Single-example posterior scatter (their Fig. 2): for ONE held-out
     target, draw many samples and scatter them in a 2D projection (two
     resonators' tuned frequencies f_t, the least-degenerate parameter
     per resonator) with the true point, posterior mean, and a 1-sigma
     confidence ellipse -- the qualitative "what does the posterior
     actually look like" picture the aggregate stats can't show.

  4. Noise-robustness check (their Fig. 4): add broadband dB noise to
     one target spectrum and compare the resulting posterior against the
     clean one, in the same 2D projection as (3).

Why thresholds/groups differ from the paper instead of copying its
numbers (the user's explicit instruction): their groups exploit their
specific setup -- 2 TVAs sharing one FIXED stiffness k, over a
[100, 350] Hz band, so f_t and mass are inversely tied and "heavy" could
be read directly off frequency (f_TVA < 120 Hz). This project has no
fixed k: m and f_t are independently sampled and k = m*(2*pi*f_t)^2 is
derived (see utils/physics.py's resonator_bounds), 3 resonators instead
of 2, and a [10, 160] Hz band. So "heavy/light" here is defined directly
on mass (the parameter it actually means), and the separation/
degeneracy thresholds are scaled to this project's narrower band rather
than reusing 15 Hz / 4 Hz outright.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from inverse_operators.common import denormalize_design, flatten_configuration, prepare_inverse_data
from inverse_operators.evaluate import MODEL_BUILDERS, SPAWN_CONTEXT, load_inverse_model, solve_configs
from inverse_operators.registry import NUM_RES
from utils.erp_dataset import configuration_to_resonators, denormalize_configuration_array, denormalize_erp_array
from utils.solver import compute_erp_spectrum

OUT_DIR = Path("plots/INVERSE_OPERATORS")
F_T_IDX = [i * 5 + 2 for i in range(NUM_RES)]  # flat-design index of each resonator's f_t
MASS_IDX = [i * 5 + 0 for i in range(NUM_RES)]  # flat-design index of each resonator's mass

WELL_SEPARATED_HZ = 10.0
NEAR_DEGENERATE_HZ = 3.0
HEAVY_KG = 0.7
LIGHT_KG = 0.3


def _difficulty_groups(true_physical: np.ndarray) -> dict[str, np.ndarray]:
    """``true_physical``: ``(n, num_res, 5)``. Returns name -> boolean mask (n,)."""
    f_t = true_physical[:, :, 2]  # (n, num_res)
    pairwise = np.abs(f_t[:, :, None] - f_t[:, None, :])  # (n, num_res, num_res)
    off_diag = pairwise + np.eye(f_t.shape[1])[None, :, :] * 1e9
    min_pairwise = off_diag.min(axis=(1, 2))
    mean_mass = true_physical[:, :, 0].mean(axis=1)
    return {
        "well_separated (min |df_t| > 10 Hz)": min_pairwise > WELL_SEPARATED_HZ,
        "near_degenerate (min |df_t| < 3 Hz)": min_pairwise < NEAR_DEGENERATE_HZ,
        "heavy (mean mass > 0.7 kg)": mean_mass > HEAVY_KG,
        "light (mean mass < 0.3 kg)": mean_mass < LIGHT_KG,
    }


def _best_of_n_physical(model_name, test_spectrum, norm, freq_hz, num_samples, pool):
    """Same best-of-N selection evaluate_design.py uses: highest log p when
    tractable, else lowest solver-checked reconstruction error. Returns
    ``(n, num_res, 5)`` physical designs.
    """
    n = test_spectrum.shape[0]
    model, _ = load_inverse_model(model_name)
    with torch.no_grad():
        result = model.sample(test_spectrum, num_samples=num_samples)
    has_log_prob = isinstance(result, tuple)
    flat_samples = result[0] if has_log_prob else result

    if has_log_prob:
        best_idx = result[1].numpy().argmax(axis=-1)
    else:
        physical_all = denormalize_design(flat_samples.numpy(), NUM_RES, norm)
        flat_physical = physical_all.reshape(n * num_samples, NUM_RES, 5)
        solved = solve_configs(pool, flat_physical, freq_hz).reshape(n, num_samples, -1)
        true_erp_db = denormalize_erp_array(test_spectrum.numpy(), norm)
        recon_mse = ((solved - true_erp_db[:, None, :]) ** 2).mean(axis=-1)
        best_idx = recon_mse.argmin(axis=-1)

    physical = denormalize_design(flat_samples.numpy(), NUM_RES, norm)
    return physical[np.arange(n), best_idx]


def stratified_recovery_table(
    model_names: list[str] | None = None,
    num_test_examples: int = 150,
    num_samples: int = 8,
    num_configurations: int = 100000,
    dataset_file=(
        "datasets/dataset_erp_ft_100k_part1.pth",
        "datasets/dataset_erp_ft_100k_part2.pth",
    ),
) -> str:
    """Table 1 analogue: MAE(m), MAE(f_t) per model, per difficulty group."""
    active_models = list(model_names) if model_names else list(MODEL_BUILDERS)
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=64,
        dataset_file=list(dataset_file), seed=727,
    )
    norm = dataset.norm_params
    freq_hz = np.asarray(dataset.frequency_values)

    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_test_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:num_test_examples]
    test_design = torch.cat(test_design, dim=0)[:num_test_examples]
    true_physical = denormalize_configuration_array(test_design.numpy(), norm)  # (n, num_res, 5)
    groups = _difficulty_groups(true_physical)

    predictions = {}
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT) as pool:
        for name in active_models:
            print(f"Stratified recovery: sampling {name} ...")
            predictions[name] = _best_of_n_physical(name, test_spectrum, norm, freq_hz, num_samples, pool)

    lines = [
        "Stratified design-parameter recovery (Dogu et al., INTER-NOISE 2026, "
        "Table 1 style -- see this file's module docstring for why the group "
        "thresholds differ from theirs)",
        "=" * 100,
    ]
    for group_name, mask in groups.items():
        n_in_group = int(mask.sum())
        lines.append(f"\n--- {group_name}  (n={n_in_group}) ---")
        if n_in_group == 0:
            lines.append("  (no test examples in this group)")
            continue
        lines.append(f"{'Model':<10} {'MAE m (kg)':>12} {'MAE f_t (Hz)':>14}")
        for name in active_models:
            pred_m = predictions[name][mask][:, :, 0]
            pred_ft = predictions[name][mask][:, :, 2]
            true_m = true_physical[mask][:, :, 0]
            true_ft = true_physical[mask][:, :, 2]
            mae_m = float(np.abs(pred_m - true_m).mean())
            mae_ft = float(np.abs(pred_ft - true_ft).mean())
            lines.append(f"{name:<10} {mae_m:>12.4g} {mae_ft:>14.4g}")

    text = "\n".join(lines)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "stratified_recovery_stats.txt").write_text(text + "\n")
    print(f"Saved {OUT_DIR / 'stratified_recovery_stats.txt'}")
    return text


def calibration_curve(
    model_names: list[str] | None = None,
    num_test_examples: int = 150,
    num_samples: int = 200,
    num_configurations: int = 100000,
    dataset_file=(
        "datasets/dataset_erp_ft_100k_part1.pth",
        "datasets/dataset_erp_ft_100k_part2.pth",
    ),
    coverage_levels=tuple(range(5, 101, 5)),
) -> None:
    """Fig. 5 analogue. Marginal per-dimension credible-interval coverage,
    no solver calls needed (containment is on the normalized flat design,
    which is a monotonic affine map of the physical one -- containment is
    identical either way).
    """
    active_models = list(model_names) if model_names else list(MODEL_BUILDERS)
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=64,
        dataset_file=list(dataset_file), seed=727,
    )
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_test_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:num_test_examples]
    test_design = torch.cat(test_design, dim=0)[:num_test_examples]
    true_flat = flatten_configuration(test_design).numpy()  # (n, 15), normalized

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 100], [0, 100], "k--", lw=1.2, label="Perfect calibration")

    for name in active_models:
        print(f"Calibration: sampling {name} ...")
        model, _ = load_inverse_model(name)
        with torch.no_grad():
            result = model.sample(test_spectrum, num_samples=num_samples)
        flat_samples = (result[0] if isinstance(result, tuple) else result).numpy()  # (n, S, 15)

        empirical = []
        for p in coverage_levels:
            lo = np.percentile(flat_samples, (100 - p) / 2, axis=1)
            hi = np.percentile(flat_samples, 100 - (100 - p) / 2, axis=1)
            inside = (true_flat >= lo) & (true_flat <= hi)  # (n, 15)
            empirical.append(100.0 * inside.mean())
        ax.plot(list(coverage_levels), empirical, marker="o", ms=3, lw=1.6, label=name)

    ax.set_xlabel("Nominal coverage (%)")
    ax.set_ylabel("Empirical coverage (%)")
    ax.set_title("Posterior calibration (marginal, per design dimension)\n"
                  "above diagonal = conservative, below = overconfident")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "calibration_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'calibration_curve.png'}")


def _scatter_with_ellipse(ax, samples_2d, true_2d, color):
    ax.scatter(samples_2d[:, 0], samples_2d[:, 1], s=6, alpha=0.25, color=color, edgecolors="none")
    mean = samples_2d.mean(axis=0)
    ax.scatter(*mean, marker="D", s=50, color="darkorange", edgecolors="black", zorder=5, label="Posterior mean")
    ax.scatter(*true_2d, marker="*", s=180, color="crimson", edgecolors="black", zorder=6, label="True")
    cov = np.cov(samples_2d.T)
    if np.all(np.isfinite(cov)) and samples_2d.shape[0] > 2:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-12, None)
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        width, height = 2 * np.sqrt(eigvals)
        ell = Ellipse(mean, width, height, angle=angle, facecolor="none", edgecolor="black", lw=1.3, zorder=4)
        ax.add_patch(ell)


def single_example_posterior(
    model_names: list[str] | None = None,
    num_test_examples: int = 150,
    num_samples: int = 1000,
    example_index: int = 0,
    num_configurations: int = 100000,
    dataset_file=(
        "datasets/dataset_erp_ft_100k_part1.pth",
        "datasets/dataset_erp_ft_100k_part2.pth",
    ),
) -> None:
    """Fig. 2 analogue: one target, all models, small multiples. Projects
    onto (f_t of resonator 1, f_t of resonator 2) -- f_t is this project's
    least-degenerate per-resonator parameter (see common.py), matching why
    the paper's own 2D plot used its two free design parameters directly.
    """
    active_models = list(model_names) if model_names else list(MODEL_BUILDERS)
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=64,
        dataset_file=list(dataset_file), seed=727,
    )
    norm = dataset.norm_params
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_test_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[example_index : example_index + 1]
    test_design = torch.cat(test_design, dim=0)[example_index : example_index + 1]
    true_physical = denormalize_configuration_array(test_design.numpy(), norm)[0]  # (num_res, 5)
    true_2d = (true_physical[0, 2], true_physical[1, 2])

    n_cols = min(3, len(active_models))
    n_rows = -(-len(active_models) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.4 * n_rows), squeeze=False)
    for idx, name in enumerate(active_models):
        print(f"Single-example posterior: sampling {name} ...")
        model, _ = load_inverse_model(name)
        with torch.no_grad():
            result = model.sample(test_spectrum, num_samples=num_samples)
        flat_samples = (result[0] if isinstance(result, tuple) else result)[0].numpy()  # (S, 15)
        physical = denormalize_design(flat_samples, NUM_RES, norm)  # (S, num_res, 5)
        samples_2d = np.stack([physical[:, 0, 2], physical[:, 1, 2]], axis=1)

        ax = axes[idx // n_cols][idx % n_cols]
        _scatter_with_ellipse(ax, samples_2d, true_2d, color="#4C72B0")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("resonator-1 f_t (Hz)")
        ax.set_ylabel("resonator-2 f_t (Hz)")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    for idx in range(len(active_models), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(
        f"Posterior samples for one held-out target ({num_samples} samples/model)\n"
        f"projected onto the two lowest-tuned resonators' f_t",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "single_example_posterior.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'single_example_posterior.png'}")


def noise_robustness_check(
    model_names: list[str] | None = None,
    num_test_examples: int = 150,
    num_samples: int = 1000,
    example_index: int = 0,
    sigma_db: float = 1.0,
    num_configurations: int = 100000,
    dataset_file=(
        "datasets/dataset_erp_ft_100k_part1.pth",
        "datasets/dataset_erp_ft_100k_part2.pth",
    ),
    seed: int = 727,
) -> None:
    """Fig. 4 analogue: clean vs. sigma_db-noisy target spectrum, same
    2D (f_t1, f_t2) posterior projection as single_example_posterior.
    """
    active_models = list(model_names) if model_names else list(MODEL_BUILDERS)
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=64,
        dataset_file=list(dataset_file), seed=727,
    )
    norm = dataset.norm_params
    freq_hz = np.asarray(dataset.frequency_values)
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_test_examples:
            break
    clean_norm_spectrum = torch.cat(test_spectrum, dim=0)[example_index : example_index + 1]
    test_design = torch.cat(test_design, dim=0)[example_index : example_index + 1]
    true_physical = denormalize_configuration_array(test_design.numpy(), norm)[0]
    true_2d = (true_physical[0, 2], true_physical[1, 2])

    clean_erp_db = denormalize_erp_array(clean_norm_spectrum.numpy(), norm)[0]
    rng = np.random.default_rng(seed)
    noisy_erp_db = clean_erp_db + rng.normal(0.0, sigma_db, size=clean_erp_db.shape)
    noisy_norm = (noisy_erp_db - norm["erp_mean"]) / norm["erp_std"]
    noisy_norm_spectrum = torch.from_numpy(noisy_norm.astype(np.float32))[None, :]

    n_cols = min(3, len(active_models))
    n_rows = -(-len(active_models) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.4 * n_rows), squeeze=False)
    for idx, name in enumerate(active_models):
        print(f"Noise robustness: sampling {name} ...")
        model, _ = load_inverse_model(name)
        with torch.no_grad():
            clean_result = model.sample(clean_norm_spectrum, num_samples=num_samples)
            noisy_result = model.sample(noisy_norm_spectrum, num_samples=num_samples)
        clean_flat = (clean_result[0] if isinstance(clean_result, tuple) else clean_result)[0].numpy()
        noisy_flat = (noisy_result[0] if isinstance(noisy_result, tuple) else noisy_result)[0].numpy()
        clean_physical = denormalize_design(clean_flat, NUM_RES, norm)
        noisy_physical = denormalize_design(noisy_flat, NUM_RES, norm)
        clean_2d = np.stack([clean_physical[:, 0, 2], clean_physical[:, 1, 2]], axis=1)
        noisy_2d = np.stack([noisy_physical[:, 0, 2], noisy_physical[:, 1, 2]], axis=1)

        ax = axes[idx // n_cols][idx % n_cols]
        ax.scatter(clean_2d[:, 0], clean_2d[:, 1], s=5, alpha=0.2, color="#4C72B0", label="Clean posterior")
        ax.scatter(noisy_2d[:, 0], noisy_2d[:, 1], s=5, alpha=0.2, color="#C44E52", label=f"Noisy posterior (+{sigma_db}dB)")
        ax.scatter(*true_2d, marker="*", s=180, color="black", edgecolors="white", zorder=6, label="True")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("resonator-1 f_t (Hz)")
        ax.set_ylabel("resonator-2 f_t (Hz)")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)
    for idx in range(len(active_models), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(
        f"Noise robustness: clean vs. +{sigma_db} dB broadband-noise target spectrum, "
        "same held-out example",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "noise_robustness.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'noise_robustness.png'}")
