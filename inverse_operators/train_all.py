"""Train all 4 trainable probabilistic inverse models and validate them.

Validation here does NOT check "did we recover the exact original design" --
the inverse problem is genuinely non-unique (see module docstrings), so that
would be the wrong test. Instead: sample several candidate designs per
target spectrum from each model, run every sampled design back through a
trained *forward* operator (the real physics-aware surrogate), and check how
well the resulting predicted spectra match the target. That is the correct
notion of "the inverse model works" for an ill-posed problem -- every
sampled design should be a plausible, spectrum-consistent solution, even if
several different samples are not identical to each other or to the
original.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from operators.operator_registry import OPERATORS
from utils.neural_operator_utils import load_operator_checkpoint
from utils.erp_dataset import denormalize_erp_array

from inverse_operators.common import prepare_inverse_data, save_checkpoint
from inverse_operators.mdn import MDN
from inverse_operators.cvae import ConditionalVAE
from inverse_operators.flow import ConditionalFlow
from inverse_operators.diffusion import ConditionalDiffusion

FORWARD_OPERATOR_KEY = "5"  # GNO -- the best forward performer, used as the validation surrogate
NUM_RES = 3
DESIGN_DIM = NUM_RES * 5


def train_one(model, loaders, loss_fn, epochs, lr, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "val": []}
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for spectrum, design in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, spectrum, design)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * spectrum.shape[0]
            n += spectrum.shape[0]
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for spectrum, design in loaders["val"]:
                loss = loss_fn(model, spectrum, design)
                total += loss.item() * spectrum.shape[0]
                n += spectrum.shape[0]
            val_loss = total / n

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"[{name}] epoch {epoch + 1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")
    return history


def main(num_configurations: int = 10000, epochs: int = 60, batch_size: int = 32):
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=batch_size,
        dataset_file="datasets/dataset_erp_ft.pth", seed=727,
    )
    norm = dataset.norm_params

    models = {
        "MDN": (MDN(design_dim=DESIGN_DIM, num_components=10), lambda m, s, d: m.training_loss(s, d), 1e-3),
        "cVAE": (ConditionalVAE(design_dim=DESIGN_DIM, latent_dim=8), lambda m, s, d: m.training_loss(s, d), 1e-3),
        "Flow": (ConditionalFlow(design_dim=DESIGN_DIM, num_layers=8, hidden=96), lambda m, s, d: m.training_loss(s, d), 5e-4),
        "Diffusion": (ConditionalDiffusion(design_dim=DESIGN_DIM, num_steps=200, hidden=128), lambda m, s, d: m.training_loss(s, d), 1e-3),
    }

    histories = {}
    for name, (model, loss_fn, lr) in models.items():
        print("\n" + "#" * 70)
        print(f"Training {name}")
        print("#" * 70)
        history = train_one(model, loaders, loss_fn, epochs=epochs, lr=lr, name=name)
        histories[name] = history
        save_checkpoint(model, norm, f"models/inverse_{name.lower()}.pth")

    # ------------------------------------------------------------------
    # Validation: sample designs, run them back through the forward
    # operator, compare reconstructed spectra to the true target.
    # ------------------------------------------------------------------
    checkpoint = load_operator_checkpoint("models/gno_erp.pth")
    fwd_spec = OPERATORS[FORWARD_OPERATOR_KEY]
    forward_model = fwd_spec["build_model"](num_res=NUM_RES, **checkpoint["model_config"])
    forward_model.load_state_dict(checkpoint["model_state_dict"])
    forward_model.eval()

    num_examples = 3
    num_samples = 6
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:num_examples]
    test_design = torch.cat(test_design, dim=0)[:num_examples]
    frequency = torch.from_numpy(
        ((dataset.frequency_values - norm["freq_mean"]) / norm["freq_std"]).astype(np.float32)
    )[None, :, None].expand(num_examples, -1, -1)

    freq_hz = np.asarray(dataset.frequency_values)
    true_erp = denormalize_erp_array(test_spectrum.numpy(), norm)

    fig, axes = plt.subplots(num_examples, len(models), figsize=(4.5 * len(models), 3.5 * num_examples))
    for row in range(num_examples):
        for col, name in enumerate(models):
            model = models[name][0]
            model.eval()
            with torch.no_grad():
                flat_samples = model.sample(test_spectrum[row : row + 1], num_samples=num_samples)[0]
            configuration = flat_samples.view(num_samples, NUM_RES, 5)
            freq_batch = frequency[row : row + 1].expand(num_samples, -1, -1)
            with torch.no_grad():
                predicted = forward_model(configuration, freq_batch)
            predicted_erp = denormalize_erp_array(predicted.numpy()[..., 0], norm)

            ax = axes[row, col] if num_examples > 1 else axes[col]
            for i in range(num_samples):
                ax.plot(freq_hz, predicted_erp[i], color="#4C72B0", alpha=0.35, lw=1.2)
            ax.plot(freq_hz, true_erp[row], color="black", lw=2, label="Target")
            if row == 0:
                ax.set_title(name, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Example {row + 1}\nERP (dB)")
            ax.set_xlabel("Frequency (Hz)")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"Inverse-design validation: {num_samples} sampled designs per method, "
        "run back through the trained forward operator (GNO)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_dir = Path("plots/INVERSE_OPERATORS")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "validation_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"Saved validation figure to {out_dir / 'validation_reconstructions.png'}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, history in histories.items():
        ax.plot(history["val"], label=name, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (method-specific NLL/MSE, not comparable across methods)")
    ax.set_title("Inverse model training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {out_dir / 'training_curves.png'}")

    print("\nDONE")


if __name__ == "__main__":
    main()
