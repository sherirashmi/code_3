"""Training script for SurrogateInverse (see surrogate_inverse.py's module
docstring for the model itself).

Not registered in erp_inverse_operators/registry.py's INVERSE_MODELS dict,
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
``erp_inverse_operators.train_all.train_one`` every other model uses; this
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

from erp_inverse_operators.common import denormalize_design, prepare_inverse_data, save_checkpoint
from erp_inverse_operators.evaluate import SPAWN_CONTEXT, solve_configs
from erp_inverse_operators.registry import DESIGN_DIM, NUM_RES
from erp_inverse_operators.surrogate_inverse import DEFAULT_SURROGATE_CHECKPOINTS, SurrogateInverse
from erp_inverse_operators.train_all import train_one
from utils.erp_dataset import denormalize_erp_array

OUT_DIR = Path("erp_inverse_operators/plots")
CHECKPOINT_PATH = "erp_inverse_operators/models/inverse_surrogate.pth"


def main(
    num_configurations: int = 10000,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 5e-4,
    surrogate_weight: float = 1.0,
    dataset_file: str = "datasets/dataset_erp_ft.pth",
    seed: int = 727,
    surrogate_checkpoints=DEFAULT_SURROGATE_CHECKPOINTS,
):
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=batch_size, dataset_file=dataset_file, seed=seed,
    )
    norm_params = dataset.norm_params

    model = SurrogateInverse(design_dim=DESIGN_DIM, surrogate_checkpoints=surrogate_checkpoints)
    print(f"SurrogateInverse params (excl. frozen surrogate ensemble): "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
          f"({model.num_surrogates} surrogate(s), deterministic point estimate)")

    def loss_fn(model, spectrum, design, epoch):
        return model.training_loss(
            spectrum,
            design,
            own_norm_params=norm_params,
            surrogate_weight=surrogate_weight,
        )

    history = train_one(model, loaders, loss_fn, epochs=epochs, lr=lr, name="SurrogateInverse")
    save_checkpoint(model, norm_params, CHECKPOINT_PATH)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train"], label="train", lw=2)
    ax.plot(history["val"], label="val", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("design_MSE + surrogate_weight * surrogate_MSE")
    ax.set_title("SurrogateInverse training curves (frozen surrogate ensemble-in-the-loop)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "surrogate_inverse_training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'surrogate_inverse_training_curves.png'}")

    _evaluate(model, dataset, loaders, norm_params)
    return model, history, dataset


def _evaluate(model, dataset, loaders, norm_params, num_examples: int = 5):
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
        predicted = model.sample(test_spectrum, num_samples=1)[:, 0, :]  # (num_examples, D)
    physical = denormalize_design(predicted.numpy(), NUM_RES, norm_params)

    solver_pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT)
    fig, axes = plt.subplots(1, num_examples, figsize=(4.8 * num_examples, 4.2))
    for row in range(num_examples):
        predicted_erp = solve_configs(solver_pool, physical[row : row + 1], freq_hz)[0]
        recon_mse = ((predicted_erp - true_erp[row]) ** 2).mean()

        ax = axes[row] if num_examples > 1 else axes
        ax.plot(freq_hz, predicted_erp, color="#C44E52", lw=2.0, label="Prediction")
        ax.plot(freq_hz, true_erp[row], color="black", lw=2, label="Target")
        ax.set_title(f"Example {row + 1} (MSE={recon_mse:.3f})", fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        if row == 0:
            ax.set_ylabel("ERP (dB)")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    solver_pool.shutdown()

    fig.suptitle(
        "SurrogateInverse validation: deterministic point estimate vs. actual solver",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_DIR / "surrogate_inverse_validation_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'surrogate_inverse_validation_reconstructions.png'}")


if __name__ == "__main__":
    main()
