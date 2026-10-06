"""Training script for SurrogateInverse (see surrogate_inverse.py's module
docstring for the model itself).

Not registered in erp_inverse/scripts/registry.py's INVERSE_MODELS dict,
for the same structural reason iFNO isn't: this model's loss needs the
CALLING dataset's own norm_params (to bridge between its own normalized
design/ERP space and each frozen surrogate's own, since they may be fit
from different dataset subsets -- see surrogate_inverse.py's module
docstring), but the registry's uniform ``build()`` is a zero-arg
lambda evaluated before any dataset is prepared, and train_all.py's
``loss_fn(model, spectrum, design, epoch)`` has no way to receive that
norm_params either. Giving this model its own small script sidesteps that
cleanly (a closure over ``dataset.norm_params`` here, no changes needed to
the shared registry/train_all.py loop other 6 models use) -- everything
else (the AdamW+cosine training loop itself) is the same
``erp_inverse.scripts.train_all.train_one`` every other model uses; this
model's training loss is a single ordinary loss function per batch, so it
needs no special multi-stage schedule the way iFNO does.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, save_checkpoint
from erp_inverse.scripts.evaluate import SPAWN_CONTEXT, solve_configs
from erp_inverse.scripts.surrogate_inverse import (
    _CONFIG_FIELDS,
    _DESIGN_PHYSICAL_BOUNDS,
    DEFAULT_SURROGATE_CHECKPOINTS,
    SurrogateInverse,
)
from erp_inverse.scripts.train_all import train_one
from utils.erp_dataset import dataset_tag_for, denormalize_erp_array
from utils.paths import inverse_model_path, inverse_plot_dir
from utils.plotting import ERP_LABEL, FREQ_LABEL, save_figure
from utils.support import device


def main(
    num_configurations: int = 100000,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 5e-4,
    surrogate_weight: float = 1.0,
    slope_weight: float = 0.5,
    peak_weight: float = 0.05,
    peak_window: int = 7,
    dataset_file=(
        "datasets/erp/3res/100k/dataset_erp_ft_100k_part1.pth",
        "datasets/erp/3res/100k/dataset_erp_ft_100k_part2.pth",
    ),
    seed: int = 727,
    surrogate_checkpoints=None,
):
    tag = dataset_tag_for(dataset_file)
    out_dir = inverse_plot_dir(tag, "Surrogate")
    if surrogate_checkpoints is None:
        from erp_inverse.scripts.surrogate_inverse import surrogate_checkpoints_for
        surrogate_checkpoints = surrogate_checkpoints_for(tag)
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=batch_size, dataset_file=dataset_file, seed=seed,
    )
    norm_params = dataset.norm_params

    num_res = int(norm_params['num_res'])
    model = SurrogateInverse(design_dim=5 * num_res, surrogate_checkpoints=surrogate_checkpoints, num_res=num_res)
    print(f"SurrogateInverse params (excl. frozen surrogate ensemble): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
          f"({model.num_surrogates} surrogate(s), deterministic point estimate)")

    def loss_fn(model, spectrum, design, epoch):
        return model.training_loss(
            spectrum,
            design,
            own_norm_params=norm_params,
            surrogate_weight=surrogate_weight,
            slope_weight=slope_weight,
            peak_weight=peak_weight,
            peak_window=peak_window,
        )

    history = train_one(model, loaders, loss_fn, epochs=epochs, lr=lr, name="SurrogateInverse")
    save_checkpoint(
        model, norm_params, inverse_model_path("Surrogate", tag),
        extra={"history": history, "dataset_tag": tag, "num_configurations": int(num_configurations),
               "modal_resolution": list(dataset.modal_resolution)},
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train"], label="train", lw=2)
    ax.plot(history["val"], label="val", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("design_MSE + surrogate_weight * surrogate_MSE")
    ax.set_title("SurrogateInverse training curves (frozen surrogate ensemble-in-the-loop)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "loss_curve.png")

    _evaluate(model, dataset, loaders, norm_params, out_dir)
    _plot_parameter_recovery(model, loaders, norm_params, out_dir)
    return model, history, dataset


def _evaluate(model, dataset, loaders, norm_params, out_dir, num_examples: int = 5):
    """Solver-validated point-estimate check -- same convention as every
    other inverse model's own validation figure (real coupled-plate
    solver, not a neural forward surrogate, scores the FINAL comparison
    even though training itself used one). Deterministic model, so there
    is exactly one prediction per target -- no "best of N samples" to pick
    from anymore.
    """
    model.eval()
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:num_examples]
    freq_hz = np.asarray(dataset.frequency_values)
    true_erp = denormalize_erp_array(test_spectrum, norm_params)

    with torch.no_grad():
        predicted = model.sample(test_spectrum.to(device), num_samples=1)[:, 0, :].cpu()  # (num_examples, D)
    physical = denormalize_design(predicted.numpy(), int(norm_params['num_res']), norm_params)

    solver_pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT)
    fig, axes = plt.subplots(1, num_examples, figsize=(4.8 * num_examples, 4.2))
    for row in range(num_examples):
        predicted_erp = solve_configs(solver_pool, physical[row : row + 1], freq_hz)[0]
        recon_mse = ((predicted_erp - true_erp[row]) ** 2).mean()

        ax = axes[row] if num_examples > 1 else axes
        ax.plot(freq_hz, predicted_erp, color="#C44E52", lw=2.0, label="Prediction")
        ax.plot(freq_hz, true_erp[row], color="black", lw=2, label="Target")
        ax.set_title(f"Example {row + 1} (MSE={recon_mse:.3f})", fontsize=10)
        ax.set_xlabel(FREQ_LABEL)
        if row == 0:
            ax.set_ylabel(ERP_LABEL)
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    solver_pool.shutdown()

    fig.suptitle(
        "SurrogateInverse validation: deterministic point estimate vs. actual solver",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_dir / "surrogate_inverse_validation_reconstructions.png")


def _plot_parameter_recovery(model, loaders, norm_params, out_dir, num_bins: int = 8, max_examples: int = 3000):
    """Predicted-vs-true design parameters, binned across the COMPLETE
    physical range each field was Latin-Hypercube-sampled within (same
    ``_DESIGN_PHYSICAL_BOUNDS`` this model soft-clamps to, not just the
    empirical min/max of whatever test examples happen to be drawn) --
    not a scatter plot, a box plot per bin, since with a deterministic
    point-estimate model the interesting question per parameter is "given
    the true value falls in this range, what's the SPREAD of predictions
    the model makes" rather than one point per example.

    Every one of up to ``max_examples`` held-out test targets contributes
    ``NUM_RES`` (m,k,f_t,x,y) tuples (flattened across resonators, same
    convention as evaluate_design.py), giving enough points per bin for a
    meaningful box plot even though there's only one prediction per
    target (no posterior to draw multiple samples from anymore).
    """
    model.eval()
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= max_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:max_examples]
    test_design = torch.cat(test_design, dim=0)[:max_examples]
    n = test_spectrum.shape[0]

    with torch.no_grad():
        predicted = model.sample(test_spectrum.to(device), num_samples=1)[:, 0, :].cpu()  # (n, D)
    predicted_physical = denormalize_design(predicted.numpy(), int(norm_params['num_res']), norm_params)  # (n, num_res, 5)
    true_physical = denormalize_design(test_design.numpy().reshape(n, -1), int(norm_params['num_res']), norm_params, consistent=False)

    fig, axes = plt.subplots(1, len(_CONFIG_FIELDS), figsize=(4.6 * len(_CONFIG_FIELDS), 4.6))
    for col, field in enumerate(_CONFIG_FIELDS):
        true_vals = true_physical[..., col].ravel()
        pred_vals = predicted_physical[..., col].ravel()

        lo, hi = _DESIGN_PHYSICAL_BOUNDS[field]  # complete dataset-generation range, not the sample's own min/max
        edges = np.linspace(lo, hi, num_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        ax = axes[col]
        box_data, positions, labels = [], [], []
        for b in range(num_bins):
            in_bin = (true_vals >= edges[b]) & (true_vals < edges[b + 1] if b < num_bins - 1 else true_vals <= edges[b + 1])
            values = pred_vals[in_bin]
            if values.size == 0:
                continue
            box_data.append(values)
            positions.append(centers[b])
            labels.append(f"{edges[b]:.2g}–{edges[b + 1]:.2g}")

        width = 0.7 * (edges[1] - edges[0])
        ax.boxplot(
            box_data, positions=positions, widths=width, showfliers=False,
            patch_artist=True, boxprops=dict(facecolor="#4C72B0", alpha=0.6),
            medianprops=dict(color="black"),
        )
        # Overrides boxplot's own (unreadable, full-float-precision) auto
        # ticks with one short label per bin EDGE, so the x-axis reads as
        # the complete range split into num_bins intervals.
        ax.set_xticks(edges)
        ax.set_xticklabels([f"{e:.3g}" for e in edges], rotation=45, ha="right", fontsize=7)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.3, label="predicted = true" if col == 0 else None)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"True {field} (complete range)")
        if col == 0:
            ax.set_ylabel("Predicted value (box = distribution per true-value bin)")
            ax.legend(fontsize=8, loc="upper left")
        ax.set_title(field, fontsize=11)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"SurrogateInverse parameter recovery: predicted value spread per true-value bin "
        f"({n} test targets x {int(norm_params['num_res'])} resonators = {n * int(norm_params['num_res'])} points/field, {num_bins} bins across each field's full generation range)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, out_dir / "surrogate_inverse_parameter_recovery.png")


if __name__ == "__main__":
    main()
