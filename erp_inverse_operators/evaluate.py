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

Design selection per target -- reported under THREE rules so models are
compared on equal terms (see ``selection_indices``):
  - random: the first i.i.d. sample (no selection at all);
  - own: the model's own target-blind rule (highest log p for the density
    models, re-scored at the physically consistent design; the point
    estimate for Surrogate; n/a for Diffusion/PadINN);
  - oracle: best of N by solver error against the target -- the same rule
    for every model, an upper bound that uses the target.
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

import functools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_inverse_operators.common import denormalize_design, prepare_inverse_data
from erp_inverse_operators.padding_inn import PadINN
from erp_inverse_operators.common import (
    BOUNDED12,
    FULL15,
    design_param_of,
    normalize_design_physical,
    sort_resonators_by_ft,
)
from erp_inverse_operators.design_space import bounded_log_abs_det
from erp_inverse_operators.registry import INVERSE_MODELS, NUM_RES, parse_variant
from utils.erp_dataset import (
    DATASETS,
    apply_model_modal_resolution,
    configuration_to_resonators,
    dataset_tag_for,
    denormalize_erp_array,
)
from utils.paths import ALL_MODELS, inverse_model_path, inverse_plot_dir
from utils.plotting import ERP_LABEL, save_figure
from utils.solver import compute_erp_spectrum
from utils.support import device

# Keyed by short name ("MDN", "cVAE", ...) rather than the registry's "1".."4"
# keys, matching this module's own reporting convention; built from the same
# INVERSE_MODELS registry train_all.py and the CLI use, so there is exactly
# one place these 4 models' constructors are defined.
#
# "Surrogate" is added on top of the registry itself (rather than inside
# INVERSE_MODELS) -- it is intentionally NOT in that registry (its training
# loss needs the calling dataset's own norm_params, see
# surrogate_inverse.py's docstring), but ITS CHECKPOINT is still just an
# ordinary state_dict + norm_params .pth like every other model's (saved by
# train_all.py's main() alongside the other 6), so it loads and evaluates
# here identically -- every constructor arg besides design_dim already
# has a usable default.
MODEL_BUILDERS = {spec["short"]: spec["build"] for spec in INVERSE_MODELS.values()}


def load_inverse_model(name: str, dataset_tag: str = "100k"):
    """Load ``models/<dataset_tag>/inverse_<name>.pth`` -> (model, norm_params)."""
    path = inverse_model_path(name, dataset_tag)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- train {name} on dataset '{dataset_tag}' first (ERP -> Inverse -> Train)."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if dataset_tag in DATASETS:
        # Solver checks of this model's designs use its training Nx x Ny.
        apply_model_modal_resolution(checkpoint, DATASETS[dataset_tag]["files"], model_name=name)
    base, spectrum_encoder, design_param = parse_variant(name)
    variant = checkpoint.get("variant", {})
    model = MODEL_BUILDERS[base](
        design_param=variant.get("design_param", design_param),
        spectrum_encoder=variant.get("spectrum_encoder", spectrum_encoder),
        num_res=int(checkpoint.get("num_res", checkpoint["norm_params"].get("num_res", NUM_RES))),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    if isinstance(model, PadINN) and "padinn_temperature" in checkpoint:
        # Rebind this instance's sample() default to the temperature
        # calibrated once at training time (train_all.py), rather than the
        # class default of 1.0 -- makes every caller that goes through this
        # loader (evaluate.py/predict.py/evaluate_design.py/
        # paper_style_report.py) use the calibrated value automatically,
        # without each needing its own PadINN special-case.
        model.sample = functools.partial(model.sample, temperature=checkpoint["padinn_temperature"])
    return model, checkpoint["norm_params"]


def checkpoint_num_configurations(name: str, dataset_tag: str) -> int | None:
    """``num_configurations`` a checkpoint was trained with (None if unrecorded)."""
    path = inverse_model_path(name, dataset_tag)
    if not path.exists():
        return None
    value = torch.load(path, map_location="cpu", weights_only=False).get("num_configurations")
    return int(value) if value is not None else None


def _check_norm(name: str, model_norm, dataset_norm) -> None:
    """Warn loudly if a model was trained with different normalisation than
    the evaluation split uses (different dataset or num_configurations) --
    its outputs would then be denormalised with the wrong statistics."""
    keys = ("erp_mean", "erp_std", "f_t_mean", "m_mean")
    if any(not np.isclose(float(model_norm[k]), float(dataset_norm[k]), rtol=1e-4) for k in keys):
        print(
            f"WARNING: {name} was trained with different normalisation than this evaluation "
            "split (different dataset or number of configurations). Re-run with the "
            "dataset/configuration count it was trained on for valid numbers."
        )


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


# --------------------------------------------------------------------------
# Scoring + the three selection rules
# --------------------------------------------------------------------------

POINT_ESTIMATE_MODELS = ("Surrogate",)


@torch.no_grad()
def density_log_prob(model, spectrum: torch.Tensor, flat: torch.Tensor) -> torch.Tensor | None:
    """``log p(design | spectrum)`` of given flat designs ``(B, S, D)`` for the
    models with a (tractable or importance-sampled) density; None otherwise
    (Diffusion, PadINN, Surrogate)."""
    from erp_inverse_operators.basis_flow import BasisFlow
    from erp_inverse_operators.cvae import ConditionalVAE
    from erp_inverse_operators.flow import ConditionalFlow
    from erp_inverse_operators.mdn import MDN

    b, s_, d = flat.shape
    if isinstance(model, MDN):
        return model._mixture_log_prob(flat, *model._params(spectrum))
    if isinstance(model, ConditionalFlow):
        cond = model.encoder(spectrum)[:, None, :].expand(-1, s_, -1).reshape(b * s_, -1)
        return model._log_prob_flat(flat.reshape(b * s_, d), cond).view(b, s_)
    if isinstance(model, (BasisFlow, ConditionalVAE)):
        return model.log_prob(spectrum, flat)
    return None


def score_samples(model, spectrum: torch.Tensor, flat_samples: torch.Tensor, norm) -> tuple[np.ndarray, np.ndarray | None]:
    """Physical designs + the density of EXACTLY those designs.

    full15: samples are projected to physically consistent designs (k
    derived, bounds clipped), then re-normalised and RE-SCORED, so the
    reported log p belongs to the design that is evaluated -- not to the raw
    pre-projection sample. bounded12: no projection exists; the density is
    converted to the physical (m, f_t, x, y) space with the transform's
    log-Jacobian.
    """
    n, s_, d = flat_samples.shape
    flat_np = flat_samples.detach().cpu().numpy()
    physical = denormalize_design(flat_np, int(norm['num_res']), norm)  # (n, S, R, 5), consistent
    if design_param_of(flat_np, int(norm['num_res'])) == BOUNDED12:
        log_p = density_log_prob(model, spectrum, flat_samples)
        if log_p is not None:
            log_p = log_p + bounded_log_abs_det(flat_samples.double(), int(norm['num_res']), norm).to(log_p.dtype)
    else:
        projected = normalize_design_physical(physical, norm, FULL15).reshape(n, s_, -1)
        log_p = density_log_prob(model, spectrum, torch.from_numpy(projected).to(spectrum.device))
    return physical, (log_p.detach().cpu().numpy() if log_p is not None else None)


def selection_indices(name: str, log_p: np.ndarray | None, solver_mse: np.ndarray) -> dict[str, np.ndarray | None]:
    """Per-target sample index under each rule.

    - ``random``: the first draw (samples are i.i.d.) -- no selection at all.
    - ``oracle``: lowest SOLVER error against the target, the same rule for
      every model; uses the target spectrum, so it is an upper bound.
    - ``own``: the model's own target-blind rule -- highest log p for density
      models, the point estimate for Surrogate, None for Diffusion/PadINN.
    """
    n = solver_mse.shape[0]
    own = None
    if parse_variant(name)[0] in POINT_ESTIMATE_MODELS:
        own = np.zeros(n, dtype=int)
    elif log_p is not None:
        own = log_p.argmax(axis=-1)
    return {"random": np.zeros(n, dtype=int), "oracle": solver_mse.argmin(axis=-1), "own": own}


def spectrum_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    error = pred - true
    per_r = [np.corrcoef(pred[i], true[i])[0, 1] for i in range(pred.shape[0])]
    ss_tot = ((true - true.mean()) ** 2).sum()
    return {
        "mae_db": float(np.abs(error).mean()),
        "rmse_db": float(np.sqrt((error**2).mean())),
        "pearson_r": float(np.corrcoef(pred.ravel(), true.ravel())[0, 1]),
        "mean_spectrum_r": float(np.nanmean(per_r)),
        "r2": float(1.0 - (error**2).sum() / ss_tot),
    }


RULE_LABELS = {
    "random": "random sample",
    "oracle": "best-of-N by solver (oracle)",
    "own": "own target-blind rule",
}


def main(
    num_test_examples: int = 150,
    num_samples: int = 8,
    num_configurations: int = 100000,
    dataset_file=(
        "datasets/dataset_erp_ft_100k_part1.pth",
        "datasets/dataset_erp_ft_100k_part2.pth",
    ),
    model_names: list[str] | None = None,
):
    """Solver-scored evaluation of the given inverse models (names may carry
    variant suffixes, e.g. ``Flow_pos_b12``) on ``num_test_examples``
    held-out targets, reported under THREE selection rules so models are
    compared on equal terms (see :func:`selection_indices`)."""
    names = list(model_names or MODEL_BUILDERS)
    dataset_tag = dataset_tag_for(dataset_file)
    missing = [n for n in names if not inverse_model_path(n, dataset_tag).exists()]
    for n in missing:
        print(f"Skipping {n}: no checkpoint at {inverse_model_path(n, dataset_tag)}")
    names = [n for n in names if n not in missing]
    if not names:
        raise FileNotFoundError(f"No trained inverse models found for dataset '{dataset_tag}'.")
    out_dir = inverse_plot_dir(dataset_tag, names[0] if model_names and len(names) == 1 else ALL_MODELS)

    stats: dict[str, dict] = {}
    predictions_db: dict[str, np.ndarray] = {}
    num_workers = max(1, os.cpu_count() or 1)
    print(f"Using {num_workers} worker processes for the physics solver.")
    data_cache: dict[str, tuple] = {}

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=SPAWN_CONTEXT) as pool:
        for name in names:
            design_param = parse_variant(name)[2]
            if design_param not in data_cache:
                dataset, loaders = prepare_inverse_data(
                    num_configurations=num_configurations, batch_size=64,
                    dataset_file=dataset_file, seed=727, design_param=design_param,
                )
                test_spectrum = []
                for spectrum, _design in loaders["test"]:
                    test_spectrum.append(spectrum)
                    if sum(t.shape[0] for t in test_spectrum) >= num_test_examples:
                        break
                test_spectrum = torch.cat(test_spectrum, dim=0)[:num_test_examples]
                data_cache[design_param] = (dataset, test_spectrum)
            dataset, test_spectrum = data_cache[design_param]
            norm = dataset.norm_params
            freq_hz = np.asarray(dataset.frequency_values)
            n = test_spectrum.shape[0]
            true_erp_db = denormalize_erp_array(test_spectrum.numpy(), norm)

            print(f"Evaluating {name} ({n} targets x {num_samples} samples, actual solver) ...")
            model, model_norm = load_inverse_model(name, dataset_tag)
            _check_norm(name, model_norm, norm)
            spectrum_dev = test_spectrum.to(device)
            with torch.no_grad():
                result = model.sample(spectrum_dev, num_samples=num_samples)
            flat_samples = (result[0] if isinstance(result, tuple) else result).to(device)
            physical, log_p = score_samples(model, spectrum_dev, flat_samples, norm)

            solved = solve_configs(pool, physical.reshape(n * num_samples, int(norm['num_res']), 5), freq_hz).reshape(n, num_samples, -1)
            solver_mse = ((solved - true_erp_db[:, None, :]) ** 2).mean(axis=-1)
            picks = selection_indices(name, log_p, solver_mse)

            stats[name] = {"point_estimate": parse_variant(name)[0] in POINT_ESTIMATE_MODELS}
            for rule, idx in picks.items():
                stats[name][rule] = None if idx is None else spectrum_metrics(solved[np.arange(n), idx], true_erp_db)
            shown = "own" if picks["own"] is not None else "random"
            stats[name]["scatter_rule"] = shown
            predictions_db[name] = solved[np.arange(n), picks[shown]]
            msg = "  ".join(
                f"{rule}: RMSE={stats[name][rule]['rmse_db']:.2f} dB" if stats[name][rule] else f"{rule}: n/a"
                for rule in ("random", "own", "oracle")
            )
            print(f"  {name}: {msg}")

    # ---- summary table ----
    lines = [
        f"Inverse-design prediction quality ({n} held-out target spectra, {num_samples} samples per target, "
        "every design scored with the actual coupled plate-resonator solver)",
        "Selection rules: random = first i.i.d. sample (no selection); own = the model's own target-blind rule",
        "(highest log p for MDN/cVAE/Flow/BasisFlow -- cVAE via importance-sampled marginal; point estimate for",
        "Surrogate; n/a for Diffusion/PadINN); oracle = best of N by solver error against the TARGET (upper bound).",
        "=" * 110,
        f"{'Model':<18} {'Rule':<8} {'MAE (dB)':>10} {'RMSE (dB)':>10} {'Pearson r':>10} {'Mean spec. r':>13} {'R^2':>8}",
        "-" * 110,
    ]
    for name, st in stats.items():
        for rule in ("random", "own", "oracle"):
            m = st[rule]
            if m is None:
                lines.append(f"{name:<18} {rule:<8} {'n/a':>10}")
                continue
            lines.append(
                f"{name:<18} {rule:<8} {m['mae_db']:>10.3f} {m['rmse_db']:>10.3f} {m['pearson_r']:>10.4f} "
                f"{m['mean_spectrum_r']:>13.4f} {m['r2']:>8.4f}"
            )
        if st["point_estimate"]:
            lines.append(f"{'':<18} (point estimator: all samples identical, so all rules coincide)")
    table_text = "\n".join(lines)
    (out_dir / "prediction_stats.txt").write_text(table_text + "\n")
    print("\n" + table_text)

    # ---- grouped bars: RMSE and R^2 under each rule ----
    names = list(stats)
    x = np.arange(len(names))
    width = 0.27
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.9 * len(names) + 4), 4.6))
    for ax, key, label in ((axes[0], "rmse_db", "RMSE (dB)"), (axes[1], "r2", r"$R^2$")):
        for j, rule in enumerate(("random", "own", "oracle")):
            vals = [stats[m][rule][key] if stats[m][rule] else np.nan for m in names]
            ax.bar(x + (j - 1) * width, vals, width, label=RULE_LABELS[rule])
        ax.set_xticks(x, names, rotation=20)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_title("Solver RMSE (lower is better)")
    axes[1].set_title(r"$R^2$ (higher is better)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Inverse design: solver-scored quality on {n} held-out spectra, three selection rules")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, out_dir / "prediction_stats_bars.png")

    # ---- prediction vs ground truth, one panel per model (own rule, else random) ----
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(names))]
    fig, axes_grid = plt.subplots(1, len(names), figsize=(4.75 * len(names), 4.8), sharex=True, sharey=True, squeeze=False)
    axes = axes_grid[0]
    lo = min(true_erp_db.min(), *(predictions_db[m].min() for m in names))
    hi = max(true_erp_db.max(), *(predictions_db[m].max() for m in names))
    rng = np.random.default_rng(0)
    for ax, name, color in zip(axes, names, colors):
        true_flat = true_erp_db.ravel()
        pred_flat = predictions_db[name].ravel()
        if true_flat.size > 6000:
            pick = rng.choice(true_flat.size, size=6000, replace=False)
            true_flat, pred_flat = true_flat[pick], pred_flat[pick]
        ax.scatter(true_flat, pred_flat, s=4, alpha=0.25, color=color, edgecolors="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label=r"$y = x$")
        m = stats[name][stats[name]["scatter_rule"]]
        ax.set_title(f"{name} ({stats[name]['scatter_rule']} rule)\nMAE $= {m['mae_db']:.2f}$ dB, $r = {m['pearson_r']:.3f}$")
        ax.set_xlabel(f"True {ERP_LABEL}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"Predicted {ERP_LABEL}\n(solver-evaluated design)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Inverse-design prediction vs. ground truth ERP (solver-scored, target-blind selection)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, out_dir / "prediction_vs_ground_truth.png")

    return stats


if __name__ == "__main__":
    main()
