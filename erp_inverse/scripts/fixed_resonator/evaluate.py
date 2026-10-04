"""Solver-scored evaluation of the position-only inverse models.

For ``num_test_examples`` held-out targets, every model draws ``num_samples``
position sets from the target's 40-120 Hz ERP. Each sampled design (positions
+ the dataset's fixed m, f_t, k) is run through the actual coupled
plate-resonator solver -- the solver that generated the dataset, not a neural
surrogate -- and compared with the target ERP. One design per target is chosen
under three rules (as in ``erp_inverse.scripts.evaluate``):

* random: the first i.i.d. sample (no selection);
* own:    the model's own target-blind pick, highest log p(positions | ERP)
          (MDN, Flow; n/a for Diffusion, which has no tractable density);
* oracle: best of N by solver error against the target in the 40-120 Hz band
          (uses the target: an upper bound).

Outputs per model, in ``plots/<dataset>/<MODEL>/``:

* ``metrics.json`` -- ERP errors (band and full 10-160 Hz) and position errors
  per rule;
* ``position_recovery.png`` -- predicted vs true x and y of each resonator;
* ``position_error_distribution.png`` -- distance between predicted and true
  resonator positions;
* ``erp_spectrum_test_config_0{1,2,3}.png`` (+ ``erp_spectrum_test_configs.png``)
  -- true vs predicted ERP next to true vs predicted positions on the plate;
* ``erp_prediction_vs_ground_truth.png`` -- predicted vs true ERP values.

``compare_models`` writes the all-model table and bar charts to ``ALL_MODELS/``.

Usage (from the repository root)::

    python -m erp_inverse.scripts.fixed_resonator.evaluate              # every trained model
    python -m erp_inverse.scripts.fixed_resonator.evaluate MDN Flow
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from erp_forward.scripts.neural_operator_utils import _configuration_features, _split_ids
from erp_inverse.scripts.evaluate import SPAWN_CONTEXT, solve_configs, spectrum_metrics
from utils.physics import Lx, Ly, edge_margin, xf, yf
from utils.plotting import ERP_LABEL, FREQ_LABEL, save_figure
from utils.support import device

from .common import ALL_MODELS, DEFAULT_DATASET, model_path, plot_dir, prepare_data
from .design_space import decode_design, position_log_abs_det, sort_by_x
from .registry import INVERSE_MODELS, SHORT_TO_KEY

RULES = ("random", "own", "oracle")
RULE_LABELS = {"random": "first sample", "own": "own pick (max. log p)", "oracle": "best of N (solver, oracle)"}
RES_COLORS = ("#2a78d6", "#eb6834", "#1f9e74", "#9467bd")
BAND_COLOR = "#e8eef8"


def load_model(name: str, dataset_tag: str):
    path = model_path(name, dataset_tag)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- train {name} first (python -m erp_inverse.scripts.fixed_resonator.train_all).")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    spec = INVERSE_MODELS[SHORT_TO_KEY[name]]
    model = spec["build"](num_res=int(checkpoint["num_res"]), spectrum_encoder=checkpoint.get("spectrum_encoder", "pooled"))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint["norm_params"]


@torch.no_grad()
def sample_designs(model, spectrum: torch.Tensor, num_samples: int, norm) -> tuple[np.ndarray, np.ndarray | None]:
    """``(n, S, R, 5)`` physical designs (sorted by x) and log p(positions | ERP)
    in physical units (``None`` for Diffusion)."""
    result = model.sample(spectrum, num_samples=num_samples)
    flat = result[0] if isinstance(result, tuple) else result
    num_res = int(norm["num_res"])
    log_p = None
    if isinstance(result, tuple):
        log_p = (result[1] + position_log_abs_det(flat, num_res, norm).to(result[1].dtype)).cpu().numpy()
    return decode_design(flat.cpu().numpy(), num_res, norm), log_p


def _pick(log_p: np.ndarray | None, band_mse: np.ndarray) -> dict[str, np.ndarray | None]:
    n = band_mse.shape[0]
    return {"random": np.zeros(n, dtype=int), "own": None if log_p is None else log_p.argmax(axis=-1),
            "oracle": band_mse.argmin(axis=-1)}


def _position_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, object]:
    """``pred`` / ``true``: (n, R, 5) physical, both sorted by x."""
    out: dict[str, object] = {}
    num_res = true.shape[1]
    for r in range(num_res):
        for name, idx in (("x", 3), ("y", 4)):
            p, t = pred[:, r, idx], true[:, r, idx]
            out[f"{name}{r + 1}"] = {"mae_cm": float(100 * np.abs(p - t).mean()),
                                     "rmse_cm": float(100 * np.sqrt(((p - t) ** 2).mean())),
                                     "pearson_r": float(np.corrcoef(p, t)[0, 1])}
        dist = 100 * np.hypot(pred[:, r, 3] - true[:, r, 3], pred[:, r, 4] - true[:, r, 4])
        out[f"distance_res{r + 1}_cm"] = {"mean": float(dist.mean()), "median": float(np.median(dist))}
    dist_all = 100 * np.hypot(pred[..., 3] - true[..., 3], pred[..., 4] - true[..., 4])
    out["distance_mean_cm"] = float(dist_all.mean())
    out["distance_median_cm"] = float(np.median(dist_all))
    for limit in (5, 10, 20):
        out[f"within_{limit}cm"] = float((dist_all <= limit).mean())
    return out


def _test_data(dataset, num_test_examples: int):
    ids = _split_ids(dataset)["test"][:num_test_examples]
    true_design = sort_by_x(_configuration_features(dataset)[ids])
    true_erp = np.asarray(dataset.responses, dtype=np.float64)[ids, :, 0]
    return ids, true_design, true_erp


def _draw_plate(ax, true_design, samples=None, picked=None, picked_label="Predicted"):
    ax.add_patch(Rectangle((0, 0), Lx, Ly, facecolor="#f4f4f1", ec="black", lw=1.1, zorder=0))
    ax.add_patch(Rectangle((edge_margin, edge_margin), Lx - 2 * edge_margin, Ly - 2 * edge_margin, fill=False,
                           ec="#9a9a94", ls="--", lw=0.8, zorder=1))
    ax.plot([xf], [yf], marker="*", color="#35d0ff", ms=14, mec="black", mew=0.8, ls="", zorder=6, label="Excitation force")
    if samples is not None:
        for r in range(true_design.shape[0]):
            ax.plot(samples[:, r, 3], samples[:, r, 4], "o", color=RES_COLORS[r], ms=3, alpha=0.35, ls="", zorder=2,
                    label="All samples" if r == 0 else None)
    for r in range(true_design.shape[0]):
        ax.plot(true_design[r, 3], true_design[r, 4], "o", color=RES_COLORS[r], ms=11, mec="black", mew=1.0, ls="",
                zorder=5, label=f"True resonator {r + 1}")
        if picked is not None:
            ax.plot(picked[r, 3], picked[r, 4], "X", color="white", ms=10, mec=RES_COLORS[r], mew=2.0, ls="", zorder=7,
                    label=f"{picked_label} resonator {r + 1}")
            ax.plot([true_design[r, 3], picked[r, 3]], [true_design[r, 4], picked[r, 4]], color=RES_COLORS[r],
                    lw=1.0, ls=":", zorder=4)
    ax.set_xlim(-0.02, Lx + 0.02)
    ax.set_ylim(-0.02, Ly + 0.02)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, Lx + 1e-9, 0.2))
    ax.set_yticks(np.arange(0, Ly + 1e-9, 0.1))
    ax.set_xlabel("Position $x$ (m)")
    ax.set_ylabel("Position $y$ (m)")


def _spectrum_example(name, i, freq, f_low, f_high, true_erp, true_design, solved, samples, own, best, own_label):
    fig, (ax, ax_plate) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.0, 1.05]})
    ax.axvspan(f_low, f_high, color=BAND_COLOR, zorder=0, label=f"ERP band seen by the model ({f_low:g}--{f_high:g} Hz)")
    for s in range(solved.shape[0]):
        ax.plot(freq, solved[s], color="0.78", lw=0.7, zorder=1, label=f"All {solved.shape[0]} samples" if s == 0 else None)
    ax.plot(freq, true_erp, color="black", lw=2.0, zorder=4, label="True ERP (target)")
    band = (freq >= f_low) & (freq <= f_high)
    rmse = lambda s: np.sqrt(((solved[s] - true_erp)[band] ** 2).mean())
    ax.plot(freq, solved[own], color="#2a78d6", lw=1.6, zorder=3, label=f"Predicted, {own_label} (band RMSE {rmse(own):.2f} dB)")
    ax.plot(freq, solved[best], color="#eb6834", lw=1.4, ls="--", zorder=3,
            label=f"Predicted, best of {solved.shape[0]} (band RMSE {rmse(best):.2f} dB)")
    ax.set_xlim(freq.min(), freq.max())
    ax.set_xlabel(FREQ_LABEL)
    ax.set_ylabel(ERP_LABEL)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title(f"Test configuration {i + 1}: ERP")
    _draw_plate(ax_plate, true_design, samples=samples, picked=samples[own], picked_label=own_label.split(" (")[0].capitalize())
    ax_plate.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8.5)
    text = "; ".join(f"R{r + 1}: true ({true_design[r, 3]:.2f}, {true_design[r, 4]:.2f}) m, "
                     f"predicted ({samples[own][r, 3]:.2f}, {samples[own][r, 4]:.2f}) m" for r in range(true_design.shape[0]))
    ax_plate.set_title(f"Resonator positions\n{text}", fontsize=9)
    fig.suptitle(f"{name}: predicted vs true ERP and resonator positions", fontsize=13)
    fig.tight_layout()
    return fig


def evaluate_model(name: str, dataset_tag: str = DEFAULT_DATASET, data=None, num_test_examples: int = 500,
                   num_samples: int = 16, num_examples: int = 3) -> dict:
    dataset, loaders = data if data is not None else prepare_data(dataset_tag)
    model, norm = load_model(name, dataset_tag)
    for key in ("pos_x_mean", "pos_y_mean", "erp_mean"):
        if not np.isclose(float(norm[key]), float(dataset.norm_params[key]), rtol=1e-5):
            print(f"WARNING: {name} was trained with different normalisation than this data split.")
    out_dir = plot_dir(dataset_tag, name)
    freq = np.asarray(dataset.frequency_values, dtype=np.float64)
    mask = dataset.band_mask
    f_low, f_high = float(norm["f_low"]), float(norm["f_high"])
    num_res = int(norm["num_res"])

    _, true_design, true_erp = _test_data(dataset, num_test_examples)
    spectra = torch.cat([s for s, _ in loaders["test"]], dim=0)[: true_design.shape[0]].to(device)
    n = spectra.shape[0]
    print(f"Evaluating {name}: {n} test targets x {num_samples} samples, every design scored with the solver ...")
    samples, log_p = sample_designs(model, spectra, num_samples, norm)  # (n, S, R, 5)
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT) as pool:
        solved = solve_configs(pool, samples.reshape(n * num_samples, num_res, 5), freq).reshape(n, num_samples, -1)
    band_mse = ((solved[..., mask] - true_erp[:, None, mask]) ** 2).mean(axis=-1)
    picks = _pick(log_p, band_mse)

    metrics: dict[str, object] = {"model": name, "dataset": dataset_tag, "num_test_examples": int(n),
                                  "num_samples": int(num_samples), "erp_band_hz": [f_low, f_high], "rules": {}}
    rows = np.arange(n)
    for rule, idx in picks.items():
        if idx is None:
            metrics["rules"][rule] = None
            continue
        chosen = solved[rows, idx]
        metrics["rules"][rule] = {
            "erp_band": spectrum_metrics(chosen[:, mask], true_erp[:, mask]),
            "erp_full": spectrum_metrics(chosen, true_erp),
            "positions": _position_metrics(samples[rows, idx], true_design),
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shown = "own" if picks["own"] is not None else "random"
    own_label = RULE_LABELS[shown]
    for rule in RULES:
        m = metrics["rules"][rule]
        if m is not None:
            print(f"  {name} [{rule}]: band ERP RMSE {m['erp_band']['rmse_db']:.2f} dB | mean position error "
                  f"{m['positions']['distance_mean_cm']:.1f} cm | within 10 cm {100 * m['positions']['within_10cm']:.0f} %")

    # ---- predicted vs true positions ----------------------------------------------
    pred = samples[rows, picks[shown]]
    fig, axes = plt.subplots(2, num_res, figsize=(5.2 * num_res, 9.0), squeeze=False)
    for r in range(num_res):
        for row, (coord, idx, hi) in enumerate((("x", 3, Lx), ("y", 4, Ly))):
            ax = axes[row, r]
            ax.scatter(true_design[:, r, idx], pred[:, r, idx], s=9, alpha=0.45, color=RES_COLORS[r], edgecolors="none")
            ax.plot([0, hi], [0, hi], "k--", lw=1.1, label="$y = x$ (perfect prediction)")
            pm = metrics["rules"][shown]["positions"][f"{coord}{r + 1}"]
            ax.set_title(f"Resonator {r + 1}, ${coord}_{r + 1}$: MAE {pm['mae_cm']:.1f} cm, $r = {pm['pearson_r']:.3f}$")
            ax.set_xlabel(f"True ${coord}_{r + 1}$ (m)")
            ax.set_ylabel(f"Predicted ${coord}_{r + 1}$ (m)")
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
            ax.set_aspect("equal")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{name}: predicted vs true resonator positions ({n} test configurations, {own_label}; "
                 "resonators numbered by increasing $x$)", fontsize=12)
    fig.tight_layout()
    save_figure(fig, out_dir / "position_recovery.png")

    # ---- position error distribution -----------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bins = np.linspace(0, 100, 51)
    for rule, color in (("random", "#9a9a94"), ("own", "#2a78d6"), ("oracle", "#eb6834")):
        if picks[rule] is None:
            continue
        chosen = samples[rows, picks[rule]]
        dist = 100 * np.hypot(chosen[..., 3] - true_design[..., 3], chosen[..., 4] - true_design[..., 4]).ravel()
        ax.hist(dist, bins=bins, histtype="step", lw=1.8, color=color,
                label=f"{RULE_LABELS[rule]}: median {np.median(dist):.1f} cm")
    ax.set_xlabel("Distance between predicted and true resonator position (cm)")
    ax.set_ylabel("Number of resonators")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"{name}: position error ({n} test configurations x {num_res} resonators)")
    fig.tight_layout()
    save_figure(fig, out_dir / "position_error_distribution.png")

    # ---- ERP + plate examples ---------------------------------------------------------
    own_pick = picks[shown]
    for i in range(min(num_examples, n)):
        fig = _spectrum_example(name, i, freq, f_low, f_high, true_erp[i], true_design[i], solved[i], samples[i],
                                int(own_pick[i]), int(picks["oracle"][i]), own_label)
        save_figure(fig, out_dir / f"erp_spectrum_test_config_{i + 1:02d}.png")
    fig, axes = plt.subplots(num_examples, 1, figsize=(10, 4.2 * num_examples), squeeze=False)
    for i, ax in enumerate(axes[:, 0]):
        ax.axvspan(f_low, f_high, color=BAND_COLOR, zorder=0)
        ax.plot(freq, true_erp[i], color="black", lw=2.0, label="True ERP (target)")
        ax.plot(freq, solved[i, own_pick[i]], color="#2a78d6", lw=1.5, label=f"Predicted, {own_label}")
        ax.plot(freq, solved[i, picks["oracle"][i]], color="#eb6834", lw=1.3, ls="--", label=f"Predicted, best of {num_samples}")
        ax.set_title(f"Test configuration {i + 1}")
        ax.set_ylabel(ERP_LABEL)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(FREQ_LABEL)
    axes[0, 0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"{name}: predicted vs true ERP (shaded: {f_low:g}--{f_high:g} Hz input band)")
    fig.tight_layout()
    save_figure(fig, out_dir / "erp_spectrum_test_configs.png")

    # ---- predicted vs true ERP values -----------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    p_vals, t_vals = solved[rows, own_pick][:, mask].ravel(), true_erp[:, mask].ravel()
    keep = np.random.default_rng(0).choice(p_vals.size, size=min(20000, p_vals.size), replace=False)
    ax.scatter(t_vals[keep], p_vals[keep], s=3, alpha=0.2, color="#2a78d6", edgecolors="none", rasterized=True)
    lo, hi = min(t_vals.min(), p_vals.min()), max(t_vals.max(), p_vals.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1, label="$y = x$")
    band = metrics["rules"][shown]["erp_band"]
    ax.set_title(f"{name} ({own_label}), {f_low:g}--{f_high:g} Hz\nMAE {band['mae_db']:.2f} dB, $r = {band['pearson_r']:.3f}$")
    ax.set_xlabel(f"True {ERP_LABEL}")
    ax.set_ylabel(f"Predicted {ERP_LABEL}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, out_dir / "erp_prediction_vs_ground_truth.png")
    return metrics


def compare_models(dataset_tag: str = DEFAULT_DATASET, names: list[str] | None = None) -> None:
    names = [n for n in (names or list(SHORT_TO_KEY)) if (plot_dir(dataset_tag, n) / "metrics.json").exists()]
    if not names:
        return
    results = {n: json.loads((plot_dir(dataset_tag, n) / "metrics.json").read_text()) for n in names}
    first = results[names[0]]
    lines = [
        f"Position-only inverse models on '{dataset_tag}': {first['num_test_examples']} test targets, "
        f"{first['num_samples']} samples each, every design scored with the solver.",
        "Rules: random = first sample; own = highest log p (MDN, Flow); oracle = best of N by solver band error (upper bound).",
        "=" * 118,
        f"{'Model':<10} {'Rule':<7} {'band RMSE':>10} {'band MAE':>9} {'band r':>7} {'full RMSE':>10} "
        f"{'pos. err mean':>14} {'median':>7} {'<=10 cm':>8} {'x MAE':>7} {'y MAE':>7} {'x r':>6} {'y r':>6}",
        f"{'':<10} {'':<7} {'(dB)':>10} {'(dB)':>9} {'':>7} {'(dB)':>10} {'(cm)':>14} {'(cm)':>7} {'':>8} {'(cm)':>7} {'(cm)':>7}",
        "-" * 118,
    ]
    for n, res in results.items():
        for rule in RULES:
            m = res["rules"].get(rule)
            if m is None:
                lines.append(f"{n:<10} {rule:<7} {'n/a':>10}")
                continue
            pos = m["positions"]
            num_res = sum(1 for k in pos if k.startswith("x") and k[1:].isdigit())
            x_mae = np.mean([pos[f"x{r + 1}"]["mae_cm"] for r in range(num_res)])
            y_mae = np.mean([pos[f"y{r + 1}"]["mae_cm"] for r in range(num_res)])
            x_r = np.mean([pos[f"x{r + 1}"]["pearson_r"] for r in range(num_res)])
            y_r = np.mean([pos[f"y{r + 1}"]["pearson_r"] for r in range(num_res)])
            lines.append(
                f"{n:<10} {rule:<7} {m['erp_band']['rmse_db']:>10.2f} {m['erp_band']['mae_db']:>9.2f} "
                f"{m['erp_band']['pearson_r']:>7.3f} {m['erp_full']['rmse_db']:>10.2f} {pos['distance_mean_cm']:>14.1f} "
                f"{pos['distance_median_cm']:>7.1f} {100 * pos['within_10cm']:>7.0f}% {x_mae:>7.1f} {y_mae:>7.1f} "
                f"{x_r:>6.3f} {y_r:>6.3f}")
    table = "\n".join(lines)
    out_dir = plot_dir(dataset_tag, ALL_MODELS)
    (out_dir / "prediction_stats.txt").write_text(table + "\n")
    print("\n" + table)

    x = np.arange(len(names))
    width = 0.27
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, getter, label in (
        (axes[0], lambda m: m["erp_band"]["rmse_db"], f"ERP RMSE, {first['erp_band_hz'][0]:g}--{first['erp_band_hz'][1]:g} Hz (dB)"),
        (axes[1], lambda m: m["positions"]["distance_mean_cm"], "Mean position error (cm)"),
    ):
        for j, rule in enumerate(RULES):
            vals = [getter(results[n]["rules"][rule]) if results[n]["rules"].get(rule) else np.nan for n in names]
            ax.bar(x + (j - 1) * width, vals, width, label=RULE_LABELS[rule])
        ax.set_xticks(x, names)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_title("Solver-scored ERP error (lower is better)")
    axes[1].set_title("Resonator position error (lower is better)")
    axes[1].legend(fontsize=8.5)
    fig.suptitle(f"Position-only inverse models: comparison on {first['num_test_examples']} test configurations")
    fig.tight_layout()
    save_figure(fig, out_dir / "prediction_stats_bars.png")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SHORT_TO_KEY)
    trained = [n for n in wanted if model_path(n, DEFAULT_DATASET).exists()]
    data = prepare_data(DEFAULT_DATASET)
    for n in trained:
        evaluate_model(n, DEFAULT_DATASET, data=data)
    compare_models(DEFAULT_DATASET, trained)
