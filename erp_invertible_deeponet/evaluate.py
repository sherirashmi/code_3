"""Evaluate the invertible DeepONet variants: forward and inverse with the same weights.

Forward: predicted vs true ERP on the test split (RMSE in dB, error at the
resonance peaks), next to the best error ANY model with Q fixed basis functions
can reach (PCA of the training spectra) -- the structural limit of the method.

Inverse: for ``num_targets`` test spectra the closed-form coefficient posterior
is sampled (``num_samples`` designs; sample 0 = posterior mean), every design is
run through the actual solver and compared with the target ERP. Rules:
own = posterior mean, random = one posterior sample, oracle = best of N by
solver error (upper bound). Design recovery: m, k, f_t, x, y per resonator
(both sorted by f_t). Inverse time per target is measured.

Saves to ``plots/<dataset>/<variant>/`` and a comparison in ``ALL_MODELS/``.

Usage (from the repository root):  python -m erp_invertible_deeponet.evaluate [Q8 Q64]
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_forward_operators.neural_operator_utils import _configuration_features, _split_ids
from erp_inverse_operators.common import denormalize_design, sort_resonators_by_ft
from erp_inverse_operators.evaluate import SPAWN_CONTEXT, solve_configs, spectrum_metrics
from utils.erp_dataset import denormalize_erp_array
from utils.plotting import ERP_LABEL, FREQ_LABEL, save_figure
from utils.support import device

from .train import DATASET, VARIANTS, load_variant, model_path, plot_dir, prepare

_draw_plate = importlib.import_module("2_res_erp_inverse_models.evaluate")._draw_plate
PARAMS = (("m", 0, "Mass $m$ (kg)"), ("k", 1, "Stiffness $k$ (kN/m)"), ("f_t", 2, "Tuning frequency $f_t$ (Hz)"),
          ("x", 3, "Position $x$ (m)"), ("y", 4, "Position $y$ (m)"))
COLORS = {"Q8": "#eb6834", "Q64": "#2a78d6"}


def pca_floor(train_db: np.ndarray, test_db: np.ndarray, q: int) -> np.ndarray:
    """Per-spectrum RMSE (dB) of the best rank-q affine reconstruction (PCA on training spectra)."""
    mu = train_db.mean(0)
    _, _, vt = np.linalg.svd(train_db - mu, full_matrices=False)
    v = vt[:q]
    y = test_db - mu
    return np.sqrt(((y - (y @ v.T) @ v) ** 2).mean(1))


def peak_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """|error| at the highest true peak of each spectrum (dB)."""
    idx = true.argmax(1)
    return np.abs(pred[np.arange(len(true)), idx] - true[np.arange(len(true)), idx])


def evaluate_variant(name, dataset, loaders, num_targets: int = 500, num_samples: int = 16, num_forward: int = 5000):
    model, norm = load_variant(name)
    out_dir = plot_dir(DATASET, name)
    freq = np.asarray(dataset.frequency_values, dtype=np.float64)
    num_res = int(dataset.num_res)
    test_ids = _split_ids(dataset)["test"]
    spectra = torch.cat([s for s, _ in loaders["test"]])
    designs = torch.cat([d for _, d in loaders["test"]])
    true_db = denormalize_erp_array(spectra.numpy(), norm).astype(np.float64)
    true_raw = sort_resonators_by_ft(_configuration_features(dataset)[test_ids].astype(np.float64))

    # ---- forward ------------------------------------------------------------------------
    with torch.no_grad():
        pred_norm = torch.cat([model(designs[i:i + 2048].reshape(-1, 4 * num_res).to(device)).cpu()
                               for i in range(0, num_forward, 2048)])[:num_forward]
    pred_db = denormalize_erp_array(pred_norm.numpy(), norm)
    f_true = true_db[:num_forward]
    fwd_rmse = np.sqrt(((pred_db - f_true) ** 2).mean(1))
    fwd_peak = peak_error(pred_db, f_true)
    train_db = denormalize_erp_array(torch.cat([s for i, (s, _) in enumerate(loaders["train"]) if i < 300]).numpy(), norm)
    floor = pca_floor(train_db, f_true, model.num_basis)

    # ---- inverse --------------------------------------------------------------------------
    target = spectra[:num_targets].to(device)
    with torch.no_grad():
        model.sample(target[:8], num_samples)  # warm-up
        t0 = time.perf_counter()
        flat = model.sample(target, num_samples)
        ms_per_target = 1000 * (time.perf_counter() - t0) / num_targets
    physical = denormalize_design(flat.cpu().numpy(), num_res, norm)  # (n, S, R, 5), bounded decode
    physical = sort_resonators_by_ft(physical.reshape(-1, num_res, 5)).reshape(num_targets, num_samples, num_res, 5)
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT) as pool:
        solved = solve_configs(pool, physical.reshape(-1, num_res, 5), freq).reshape(num_targets, num_samples, -1)
    t_db = true_db[:num_targets]
    mse = ((solved - t_db[:, None]) ** 2).mean(-1)
    picks = {"own": np.zeros(num_targets, int), "random": np.ones(num_targets, int), "oracle": mse.argmin(1)}
    rows = np.arange(num_targets)
    truth = true_raw[:num_targets]
    metrics = {"variant": name, "Q": model.num_basis, "D": model.design_dim, "num_targets": num_targets,
               "num_samples": num_samples, "inverse_ms_per_target": ms_per_target,
               "forward": {"rmse_median_db": float(np.median(fwd_rmse)), "rmse_mean_db": float(fwd_rmse.mean()),
                           "peak_error_median_db": float(np.median(fwd_peak)),
                           "pca_floor_median_db": float(np.median(floor))},
               "inverse": {}}
    for rule, idx in picks.items():
        chosen = physical[rows, idx]
        rec = {}
        for pname, j, _ in PARAMS:
            p, t = chosen[..., j].ravel(), truth[..., j].ravel()
            rec[pname] = {"mae": float(np.abs(p - t).mean()), "pearson_r": float(np.corrcoef(p, t)[0, 1])}
        dist = 100 * np.hypot(chosen[..., 3] - truth[..., 3], chosen[..., 4] - truth[..., 4])
        rec["position_error_median_cm"] = float(np.median(dist))
        metrics["inverse"][rule] = {"erp": spectrum_metrics(solved[rows, idx], t_db), "design": rec}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"iDON-{name}: forward RMSE median {np.median(fwd_rmse):.2f} dB (PCA floor {np.median(floor):.2f} dB), "
          f"peak error {np.median(fwd_peak):.1f} dB | inverse ERP RMSE own {metrics['inverse']['own']['erp']['rmse_db']:.2f} / "
          f"oracle {metrics['inverse']['oracle']['erp']['rmse_db']:.2f} dB | {ms_per_target:.2f} ms per target")

    # ---- plots -----------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, i in zip(axes, (0, 1, 2)):
        ax.plot(freq, f_true[i], color="black", lw=2.0, label="True ERP (solver)")
        ax.plot(freq, pred_db[i], color=COLORS.get(name, "#2a78d6"), lw=1.4, ls="--",
                label=f"Invertible DeepONet {name} (RMSE {fwd_rmse[i]:.2f} dB)")
        ax.set_ylabel(ERP_LABEL)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
        ax.set_title(f"Test configuration {i + 1}", fontsize=10)
    axes[-1].set_xlabel(FREQ_LABEL)
    fig.suptitle(f"Invertible DeepONet {name} ($Q = {model.num_basis}$): forward prediction", fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir / "forward_examples.png")

    chosen = physical[rows, picks["own"]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (pname, j, label) in zip(axes.ravel(), PARAMS):
        scale = 1e-3 if pname == "k" else 1.0
        p, t = chosen[..., j].ravel() * scale, truth[..., j].ravel() * scale
        ax.scatter(t, p, s=6, alpha=0.4, color=COLORS.get(name, "#2a78d6"), edgecolors="none")
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        r = metrics["inverse"]["own"]["design"][pname]
        ax.set_title(f"{label}: $r = {r['pearson_r']:.3f}$", fontsize=10)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted (posterior mean)")
        ax.grid(alpha=0.3)
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"Invertible DeepONet {name}: predicted vs true design ({num_targets} test spectra, both resonators)", fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir / "design_recovery.png")

    for k in range(3):
        fig, (ax, axp) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.0, 1.05]})
        for s in range(num_samples):
            ax.plot(freq, solved[k, s], color="0.8", lw=0.6, label=f"All {num_samples} posterior samples" if s == 0 else None)
        ax.plot(freq, t_db[k], color="black", lw=2.0, label="Target ERP")
        ax.plot(freq, solved[k, 0], color="#2a78d6", lw=1.5, label=f"Posterior mean (RMSE {np.sqrt(mse[k, 0]):.2f} dB)")
        ax.plot(freq, solved[k, picks['oracle'][k]], color="#eb6834", lw=1.3, ls="--",
                label=f"Best of {num_samples} (RMSE {np.sqrt(mse[k, picks['oracle'][k]]):.2f} dB)")
        ax.set_xlabel(FREQ_LABEL)
        ax.set_ylabel(ERP_LABEL)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="lower right")
        ax.set_title("ERP of the designs (checked with the solver)")
        _draw_plate(axp, truth[k], samples=physical[k], picked=physical[k, 0], picked_label="Posterior mean")
        axp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
        tx = "; ".join(f"R{r + 1}: $m$ {truth[k, r, 0]:.2f}/{physical[k, 0, r, 0]:.2f} kg, $f_t$ {truth[k, r, 2]:.0f}/{physical[k, 0, r, 2]:.0f} Hz"
                       for r in range(num_res))
        axp.set_title(f"Positions (filled: true)\ntrue/predicted: {tx}", fontsize=9)
        fig.suptitle(f"Invertible DeepONet {name}: inverse design of test configuration {k + 1}", fontsize=13)
        fig.tight_layout()
        save_figure(fig, out_dir / f"inverse_example_{k + 1}.png")
    return metrics


def main(names=None):
    names = [n for n in (names or list(VARIANTS)) if model_path(n).exists()]
    dataset, loaders = prepare()
    results = {n: evaluate_variant(n, dataset, loaders) for n in names}
    lines = [f"Invertible DeepONet on '{DATASET}' (forward on 5,000 and inverse on 500 test configurations)", "",
             f"{'Variant':<8} {'Q':>4} {'fwd RMSE':>9} {'PCA floor':>10} {'peak err':>9} | {'inv own':>8} {'inv best':>9} "
             f"{'f_t r':>6} {'m r':>6} {'x r':>6} {'y r':>6} {'pos err':>8} {'ms/target':>10}",
             f"{'':<8} {'':>4} {'(dB)':>9} {'(dB)':>10} {'(dB)':>9} | {'(dB)':>8} {'(dB)':>9} {'':>6} {'':>6} {'':>6} {'':>6} {'(cm)':>8}"]
    for n, m in results.items():
        own, best = m["inverse"]["own"], m["inverse"]["oracle"]
        d = own["design"]
        lines.append(f"{n:<8} {m['Q']:>4} {m['forward']['rmse_median_db']:>9.2f} {m['forward']['pca_floor_median_db']:>10.2f} "
                     f"{m['forward']['peak_error_median_db']:>9.1f} | {own['erp']['rmse_db']:>8.2f} {best['erp']['rmse_db']:>9.2f} "
                     f"{d['f_t']['pearson_r']:>6.3f} {d['m']['pearson_r']:>6.3f} {d['x']['pearson_r']:>6.3f} {d['y']['pearson_r']:>6.3f} "
                     f"{d['position_error_median_cm']:>8.1f} {m['inverse_ms_per_target']:>10.2f}")
    text = "\n".join(lines)
    (plot_dir(DATASET) / "comparison.txt").write_text(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
