"""Predict with one inverse model: target ERP spectrum -> sampled designs.

Mirrors forward_operators' "predict" action (one input in, one prediction
out, with a saved comparison plot), but for the inverse direction: the
"input" is a target ERP spectrum and the "prediction" is a small set of
candidate resonator configurations, each forward-checked through the
actual solver and scored the same way evaluate.py/train_all.py score
samples (log-probability where tractable, spectrum-consistency for
Diffusion).

The target spectrum itself can come from either a resonator configuration
you supply (solved for real, so it's a genuine physical target -- typing
301 raw dB numbers by hand isn't practical) or the first spectrum from the
saved test split, matching forward_operators' own two prediction-input
choices.
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

from inverse_operators.common import denormalize_design, prepare_inverse_data
from inverse_operators.evaluate import SPAWN_CONTEXT, load_inverse_model, solve_configs
from inverse_operators.registry import NUM_RES
from inverse_operators.train_all import format_configuration
from utils.erp_dataset import (
    configuration_to_resonators,
    denormalize_configuration_array,
    denormalize_erp_array,
)
from utils.solver import compute_erp_spectrum

OUT_DIR = Path("plots/INVERSE_OPERATORS")


def predict_one(
    key_short: str,
    *,
    configuration: np.ndarray | None = None,
    num_samples: int = 6,
    num_configurations: int = 10000,
    dataset_file="datasets/dataset_erp_ft.pth",
    seed: int = 727,
):
    """Sample ``num_samples`` candidate designs for one target spectrum.

    ``configuration``: a real ``(num_res, 5)`` physical ``[m,k,f_t,x,y]``
    configuration -- its actual solver-computed ERP becomes the target to
    invert. ``None`` uses the first spectrum from the saved test split
    instead (its own true configuration is not shown to the model or used
    anywhere except as informational context in the printed report -- the
    inverse problem is non-unique, so "recovering" that specific design is
    not the goal, see inverse_operators/common.py).

    Returns a dict with the target spectrum, sampled physical designs,
    scores/confidences, the best index, and the saved plot path.
    """
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=64,
        dataset_file=dataset_file, seed=seed,
    )
    norm = dataset.norm_params
    freq_hz = np.asarray(dataset.frequency_values)

    true_configuration = None
    if configuration is not None:
        resonators = configuration_to_resonators(np.asarray(configuration, dtype=np.float64))
        target_erp_db = compute_erp_spectrum(resonators, frequencies=freq_hz)
    else:
        for spectrum, design in loaders["test"]:
            target_erp_db = denormalize_erp_array(spectrum[:1].numpy(), norm)[0]
            # design is already (batch, num_res, 5), not flat -- unlike a
            # model's sampled output, so this uses the array-shaped
            # denormalizer directly instead of denormalize_design (which
            # expects a flat (..., num_res*5) vector).
            true_configuration = denormalize_configuration_array(design[:1].numpy(), norm)[0]
            break

    target_norm = (target_erp_db - norm["erp_mean"]) / norm["erp_std"]
    target_spectrum = torch.from_numpy(target_norm.astype(np.float32))[None, :]

    model, _ = load_inverse_model(key_short)
    model.eval()
    with torch.no_grad():
        result = model.sample(target_spectrum, num_samples=num_samples)
    has_log_prob = isinstance(result, tuple)
    flat_samples = result[0][0] if has_log_prob else result[0]
    log_probs = result[1][0] if has_log_prob else None

    physical = denormalize_design(flat_samples.numpy(), NUM_RES, norm)  # (num_samples, num_res, 5)

    pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT)
    try:
        predicted_erp = solve_configs(pool, physical, freq_hz)  # (num_samples, n_freq), real dB
    finally:
        pool.shutdown()
    recon_mse = ((predicted_erp - target_erp_db[None, :]) ** 2).mean(axis=1)

    if has_log_prob:
        log_probs_np = log_probs.numpy()
        confidence = np.exp(log_probs_np - log_probs_np.max())
        confidence = confidence / confidence.sum()
        best_idx = int(log_probs_np.argmax())
        score_label = "log p(design|spectrum)"
        scores = log_probs_np
    else:
        confidence = np.exp(-recon_mse) / np.exp(-recon_mse).sum()
        best_idx = int(recon_mse.argmin())
        score_label = "spectrum-consistency (-MSE, NOT a probability)"
        scores = -recon_mse

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i in range(num_samples):
        if i == best_idx:
            continue
        ax.plot(freq_hz, predicted_erp[i], color="#4C72B0", alpha=0.3, lw=1.1)
    ax.plot(freq_hz, predicted_erp[best_idx], color="#C44E52", lw=2.0, label="Best sample")
    ax.plot(freq_hz, target_erp_db, color="black", lw=2, label="Target")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ERP (dB)")
    ax.set_title(f"{key_short}: {num_samples} sampled designs vs. target ({score_label})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = OUT_DIR / f"predict_{key_short.lower()}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "target_erp_db": target_erp_db,
        "true_configuration": true_configuration,
        "physical_designs": physical,
        "predicted_erp_db": predicted_erp,
        "scores": scores,
        "score_label": score_label,
        "confidence": confidence,
        "best_idx": best_idx,
        "plot_path": out_path,
    }


def print_report(key_short: str, result: dict) -> None:
    print(f"\n{key_short} sampled {result['physical_designs'].shape[0]} candidate designs "
          f"({result['score_label']}):")
    if result["true_configuration"] is not None:
        print("(Target came from a real test-split configuration -- shown for reference only; "
              "the inverse problem is non-unique, so a different design can still be a valid match.)")
        print("True configuration:")
        print(format_configuration(result["true_configuration"]))
    for i in range(result["physical_designs"].shape[0]):
        marker = " <-- best" if i == result["best_idx"] else ""
        print(
            f"sample {i + 1}: score={result['scores'][i]:+.4f}  "
            f"relative_confidence={result['confidence'][i] * 100:5.1f}%{marker}"
        )
        print(format_configuration(result["physical_designs"][i]))
    print(f"Saved prediction plot to {result['plot_path']}")
