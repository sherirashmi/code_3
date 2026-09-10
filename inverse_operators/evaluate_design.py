"""Predicted vs. true resonator design parameters (m, k, x, y) for the 4 inverse models.

Important caveat, not a technicality: the inverse problem is genuinely
non-unique (see inverse_operators/common.py's module docstring) -- several
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

Design selection per target matches evaluate.py: highest log p(design |
spectrum) for MDN/cVAE/Flow (blind to the target, no solver needed at all);
lowest solver-simulated reconstruction error for Diffusion (the only
score it has).
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from inverse_operators.common import denormalize_design, prepare_inverse_data
from inverse_operators.evaluate import (
    MODEL_BUILDERS,
    SPAWN_CONTEXT,
    load_inverse_model,
    solve_configs,
)
from utils.erp_dataset import denormalize_configuration_array, denormalize_erp_array
from concurrent.futures import ProcessPoolExecutor

NUM_RES = 3
OUT_DIR = Path("plots/INVERSE_OPERATORS")
PARAMS = [
    ("m", 0, "Mass (kg)"),
    ("k", 1, "Spring constant (N/m)"),
    ("x", 3, "x position (m)"),
    ("y", 4, "y position (m)"),
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
    n = test_spectrum.shape[0]

    # True design in physical units -- already canonicalized (sorted by
    # ascending f_t) by InverseDesignDataset, matching the order every model
    # was trained to predict, so predicted[i, r] and true[i, r] refer to
    # "the r-th resonator by ascending tuned frequency", not an arbitrary
    # label -- no permutation ambiguity left to resolve here.
    true_physical = denormalize_configuration_array(test_design.numpy(), norm)  # (n, num_res, 5)

    predictions = {}
    num_workers = max(1, os.cpu_count() or 1)

    with ProcessPoolExecutor(max_workers=num_workers, mp_context=SPAWN_CONTEXT) as pool:
        for name in active_models:
            print(f"Evaluating {name} design-parameter recovery ...")
            model, _ = load_inverse_model(name)
            with torch.no_grad():
                result = model.sample(test_spectrum, num_samples=num_samples)
            has_log_prob = isinstance(result, tuple)
            flat_samples = result[0] if has_log_prob else result  # (n, num_samples, design_dim)

            if has_log_prob:
                log_probs = result[1].numpy()
                best_idx = log_probs.argmax(axis=-1)
            else:
                # Diffusion: no tractable density, so its own target-blind
                # selection doesn't exist -- fall back to the solver, same
                # as evaluate.py.
                physical_all = denormalize_design(flat_samples.numpy(), NUM_RES, norm)
                flat_physical = physical_all.reshape(n * num_samples, NUM_RES, 5)
                solved = solve_configs(pool, flat_physical, freq_hz).reshape(n, num_samples, -1)
                true_erp_db = denormalize_erp_array(test_spectrum.numpy(), norm)
                recon_mse = ((solved - true_erp_db[:, None, :]) ** 2).mean(axis=-1)
                best_idx = recon_mse.argmin(axis=-1)

            physical = denormalize_design(flat_samples.numpy(), NUM_RES, norm)  # (n, num_samples, num_res, 5)
            best_physical = physical[np.arange(n), best_idx]  # (n, num_res, 5)
            predictions[name] = best_physical

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = active_models
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    stats = {name: {} for name in names}
    lines = [
        f"Design-parameter recovery ({n} held-out targets x {NUM_RES} resonators = "
        f"{n * NUM_RES} points per model/parameter)",
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
            ax.set_title(f"{name}\nMAE={mae:.3g}, r={r:.3f}", fontsize=10)
            if row == len(PARAMS) - 1:
                ax.set_xlabel(f"True {label}")
            if col == 0:
                ax.set_ylabel(f"Predicted {label}")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        "Inverse-design parameter recovery: predicted vs. true [m, k, x, y]\n"
        "(non-unique problem -- low correlation can mean a different valid design, not a wrong one)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "design_parameter_recovery.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'design_parameter_recovery.png'}")

    table_text = "\n".join(lines)
    (OUT_DIR / "design_parameter_stats.txt").write_text(table_text + "\n")
    print("\n" + table_text)

    return stats


if __name__ == "__main__":
    main()
