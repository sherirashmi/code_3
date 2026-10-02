"""Predicted vs. true resonator design parameters (m, k, x, y) for the 4 inverse models.

Important caveat, not a technicality: the inverse problem is genuinely
non-unique (see erp_inverse_operators/common.py's module docstring) -- several
different [m,k,x,y] configurations can produce very similar ERP spectra, most
directly through the mass/stiffness degeneracy (f_t = sqrt(k/m)/(2*pi) is what
mostly sets a resonator's effect on the spectrum, so many (m,k) pairs sharing
one f_t are close to interchangeable). A model can therefore score well on
the spectrum-level comparison (evaluate.py) while still disagreeing with the
*specific* m/k values used to generate a given target -- that is not
necessarily an error, it can be a different-but-valid solution. This script
reports the comparison anyway because it is still informative (e.g. whether
position, which is far less degenerate, is recovered better than m/k), but
the numbers should be read as "how close to the original recipe", not
"how wrong the model is".

Design selection per target is TARGET-BLIND for every model: the model's
own rule (highest log p, re-scored at the physically consistent design, for
MDN/cVAE/Flow/BasisFlow; the point estimate for Surrogate), otherwise the
first i.i.d. sample (Diffusion, PadINN). Predicted resonators are sorted by
their predicted f_t before the slot-by-slot comparison with the (f_t-sorted)
true design.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_inverse_operators.common import denormalize_design, prepare_inverse_data
from erp_inverse_operators.common import sort_resonators_by_ft
from erp_inverse_operators.evaluate import (
    MODEL_BUILDERS,
    _check_norm,
    load_inverse_model,
    score_samples,
    selection_indices,
)
from erp_inverse_operators.registry import parse_variant
from utils.erp_dataset import dataset_tag_for, denormalize_configuration_array, denormalize_erp_array
from utils.paths import ALL_MODELS, inverse_model_path, inverse_plot_dir
from utils.plotting import save_figure
from utils.support import device
from concurrent.futures import ProcessPoolExecutor

PARAMS = [
    ("m", 0, r"mass $m$ (kg)"),
    ("k", 1, r"stiffness $k$ (N/m)"),
    ("f_t", 2, r"tuning frequency $f_t$ (Hz)"),
    ("x", 3, r"position $x$ (m)"),
    ("y", 4, r"position $y$ (m)"),
]


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
    """``model_names`` restricts this to a subset of MODEL_BUILDERS (default
    all), same rationale as evaluate.main()'s own ``model_names`` param.
    """
    dataset_tag = dataset_tag_for(dataset_file)
    active_models = [
        n for n in (list(model_names) if model_names else list(MODEL_BUILDERS))
        if inverse_model_path(n, dataset_tag).exists() or print(f"Skipping {n}: not trained on '{dataset_tag}'.")
    ]
    if not active_models:
        raise FileNotFoundError(f"No trained inverse models found for dataset '{dataset_tag}'.")
    out_dir = inverse_plot_dir(dataset_tag, active_models[0] if model_names and len(active_models) == 1 else ALL_MODELS)

    # Target-blind selection only (no solver, no target): each model's own
    # rule where it has one (highest log p -- re-scored at the physically
    # consistent design -- or the Surrogate point estimate), otherwise the
    # first i.i.d. sample. Using the solver here (as before, for Diffusion)
    # would let some models pick with knowledge of the target and others not.
    predictions, rules = {}, {}
    data_cache: dict[str, tuple] = {}
    true_physical = None
    for name in active_models:
        design_param = parse_variant(name)[2]
        if design_param not in data_cache:
            dataset, loaders = prepare_inverse_data(
                num_configurations=num_configurations, batch_size=64,
                dataset_file=dataset_file, seed=727, design_param=design_param,
            )
            test_spectrum, test_design = [], []
            for spectrum, design in loaders["test"]:
                test_spectrum.append(spectrum)
                test_design.append(design)
                if sum(t.shape[0] for t in test_spectrum) >= num_test_examples:
                    break
            test_spectrum = torch.cat(test_spectrum, dim=0)[:num_test_examples]
            test_design = torch.cat(test_design, dim=0)[:num_test_examples]
            data_cache[design_param] = (dataset, test_spectrum, test_design)
        dataset, test_spectrum, test_design = data_cache[design_param]
        norm = dataset.norm_params
        n = test_spectrum.shape[0]
        if true_physical is None:
            # Canonical (ascending f_t) true designs, identical for every
            # design parameterisation (same split, exact decode).
            true_physical = denormalize_design(test_design.reshape(n, -1).numpy(), int(norm['num_res']), norm, consistent=False)

        print(f"Evaluating {name} design-parameter recovery ...")
        model, model_norm = load_inverse_model(name, dataset_tag)
        _check_norm(name, model_norm, norm)
        spectrum_dev = test_spectrum.to(device)
        with torch.no_grad():
            result = model.sample(spectrum_dev, num_samples=num_samples)
        flat_samples = (result[0] if isinstance(result, tuple) else result).to(device)
        physical, log_p = score_samples(model, spectrum_dev, flat_samples, norm)
        own = selection_indices(name, log_p, np.zeros((n, num_samples)))["own"]
        rules[name] = "own rule" if own is not None else "random sample"
        idx = own if own is not None else np.zeros(n, dtype=int)
        # Sort the PREDICTED resonators by their own predicted f_t before
        # comparing slot-by-slot with the (f_t-sorted) truth -- the solver
        # score is order-independent, parameter recovery is not.
        predictions[name] = sort_resonators_by_ft(physical[np.arange(n), idx])

    names = active_models
    # tab10 scales to any number of models automatically -- this literal
    # list has needed a manual bump every time a model was added (4 -> 5 -> 6).
    colors = [plt.get_cmap("tab10")(i) for i in range(len(names))]

    stats = {name: {} for name in names}
    lines = [
        f"Design-parameter recovery ({n} held-out targets x {int(norm['num_res'])} resonators = "
        f"{n * int(norm['num_res'])} points per model/parameter)",
        "Caveat: the inverse problem is non-unique -- see this file's module docstring.",
        "=" * 90,
    ]

    # squeeze=False: a 1-column grid (the CLI's single-model workflow)
    # would otherwise collapse to a 1D array and break axes[row, col].
    fig, axes = plt.subplots(
        len(PARAMS), len(names), figsize=(4.4 * len(names), 4.0 * len(PARAMS)), squeeze=False
    )
    for row, (short, idx, label) in enumerate(PARAMS):
        true_vals = true_physical[..., idx].ravel()  # (n*num_res,)
        lo, hi = true_vals.min(), true_vals.max()
        lines.append(f"\n--- {label} ---")
        lines.append(f"{'Model':<10} {'MAE':>14} {'Pearson r':>10}")
        for col, name in enumerate(names):
            pred_vals = predictions[name][..., idx].ravel()
            mae = float(np.abs(pred_vals - true_vals).mean())
            r = float(np.corrcoef(pred_vals, true_vals)[0, 1])
            stats[name][short] = {"mae": mae, "pearson_r": r}
            lines.append(f"{name:<10} {mae:>14.4g} {r:>10.4f}")

            ax = axes[row, col]
            ax.scatter(true_vals, pred_vals, s=8, alpha=0.35, color=colors[col], edgecolors="none")
            span = [min(lo, pred_vals.min()), max(hi, pred_vals.max())]
            ax.plot(span, span, "k--", lw=1.1, label="y = x" if row == 0 and col == 0 else None)
            ax.set_title(f"{name} ({rules[name]})\nMAE $= {mae:.3g}$, $r = {r:.3f}$", fontsize=10)
            if row == len(PARAMS) - 1:
                ax.set_xlabel(f"True {label}")
            if col == 0:
                ax.set_ylabel(f"Predicted {label}")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Inverse-design parameter recovery: predicted vs. true $[m, k, f_t, x, y]$\n"
        "(non-unique problem -- low correlation can mean a different valid design, not a wrong one)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_dir / "design_parameter_recovery.png")

    table_text = "\n".join(lines)
    (out_dir / "design_parameter_stats.txt").write_text(table_text + "\n")
    print("\n" + table_text)

    return stats


if __name__ == "__main__":
    main()
