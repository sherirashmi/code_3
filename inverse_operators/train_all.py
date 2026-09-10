"""Train all 4 trainable probabilistic inverse models and validate them.

Validation here does NOT check "did we recover the exact original design" --
the inverse problem is genuinely non-unique (see module docstrings), so that
would be the wrong test. Instead: sample several candidate designs per
target spectrum from each model, run every sampled design through the
*actual coupled plate-resonator solver* (``utils.solver.compute_erp_spectrum``
-- the same modal solver that generated the dataset, not a neural forward
surrogate), and check how well the resulting predicted spectra match the
target. Using the real solver instead of a forward operator removes that
operator's own approximation error from the inverse-model score entirely.

Training fixes applied here (see the per-method module docstrings for why):
  - Cosine LR annealing for every model (was a flat LR for all 80 epochs).
  - Best-validation-epoch checkpointing (was: whatever the last epoch
    produced, which silently kept MDN's overfit epoch-80 weights).
  - KL annealing for the cVAE (beta ramps up over the first few epochs
    instead of being fixed from step 1, which is the classic cause of
    posterior collapse -- and this model showed exactly that signature:
    a flat loss curve and near-zero sample-to-sample diversity).
  - Cosine noise schedule + fewer steps for the diffusion model (see
    diffusion.py's docstring).

Per-sample reporting: for MDN, Flow, and cVAE -- the three methods with a
tractable ``log p(design | spectrum)`` -- every sampled design is reported
with its exact log-probability under the model, plus a softmax-normalized
"relative confidence" across that method's own samples for the same target
(these sum to 1 and are the more directly interpretable number: "how much
more plausible is this candidate than the others sampled for the same
target", not an absolute probability, which a density value alone isn't).
Diffusion has no tractable density with this simple formulation; it's
reported instead with a spectrum-consistency score (how well the sampled
design's solver-simulated spectrum matches the target) -- explicitly
labeled as such, not as a probability, so the two aren't confused.
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.erp_dataset import denormalize_erp_array

from inverse_operators.common import prepare_inverse_data, save_checkpoint, denormalize_design
from inverse_operators.evaluate import SPAWN_CONTEXT, solve_configs
from inverse_operators.mdn import MDN
from inverse_operators.cvae import ConditionalVAE
from inverse_operators.flow import ConditionalFlow
from inverse_operators.diffusion import ConditionalDiffusion

NUM_RES = 3
DESIGN_DIM = NUM_RES * 5
KL_WARMUP_EPOCHS = 30
KL_TARGET_BETA = 0.1


def train_one(model, loaders, loss_fn, epochs, lr, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for spectrum, design in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, spectrum, design, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * spectrum.shape[0]
            n += spectrum.shape[0]
        train_loss = total / n
        scheduler.step()

        model.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for spectrum, design in loaders["val"]:
                loss = loss_fn(model, spectrum, design, epoch)
                total += loss.item() * spectrum.shape[0]
                n += spectrum.shape[0]
            val_loss = total / n

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{name}] epoch {epoch + 1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{name}] restored best-validation checkpoint (val={best_val:.4f})")
    return history


def format_configuration(configuration: np.ndarray) -> str:
    lines = []
    for i, (m, k, f_t, x, y) in enumerate(configuration):
        lines.append(f"  res{i + 1}: m={m:.3f}kg k={k:,.0f}N/m f_t={f_t:.1f}Hz x={x:.3f}m y={y:.3f}m")
    return "\n".join(lines)


def main(
    num_configurations: int = 10000,
    epochs: int = 150,
    batch_size: int = 64,
    dataset_file="datasets/dataset_erp_ft.pth",
):
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=batch_size,
        dataset_file=dataset_file, seed=727,
    )
    norm = dataset.norm_params

    def mdn_loss(m, s, d, epoch):
        return m.training_loss(s, d)

    def cvae_loss(m, s, d, epoch):
        beta = KL_TARGET_BETA * min(1.0, epoch / KL_WARMUP_EPOCHS)
        return m.training_loss(s, d, beta=beta)

    def flow_loss(m, s, d, epoch):
        return m.training_loss(s, d)

    def diffusion_loss(m, s, d, epoch):
        return m.training_loss(s, d)

    models = {
        "MDN": (MDN(design_dim=DESIGN_DIM, num_components=10), mdn_loss, 1e-3),
        "cVAE": (ConditionalVAE(design_dim=DESIGN_DIM, latent_dim=8), cvae_loss, 1e-3),
        "Flow": (ConditionalFlow(design_dim=DESIGN_DIM, num_layers=8, hidden=96), flow_loss, 5e-4),
        "Diffusion": (ConditionalDiffusion(design_dim=DESIGN_DIM, num_steps=100, hidden=128), diffusion_loss, 1e-3),
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
    # Validation + per-sample reporting.
    # ------------------------------------------------------------------
    solver_pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT)

    num_examples = 3
    num_samples = 6
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:num_examples]
    frequency = torch.from_numpy(
        ((dataset.frequency_values - norm["freq_mean"]) / norm["freq_std"]).astype(np.float32)
    )[None, :, None].expand(num_examples, -1, -1)

    freq_hz = np.asarray(dataset.frequency_values)
    true_erp = denormalize_erp_array(test_spectrum.numpy(), norm)

    out_dir = Path("plots/INVERSE_OPERATORS")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = []

    fig, axes = plt.subplots(num_examples, len(models), figsize=(4.8 * len(models), 4.2 * num_examples))
    for row in range(num_examples):
        report_lines.append(f"\n{'=' * 90}\nExample {row + 1}\n{'=' * 90}")
        for col, name in enumerate(models):
            model = models[name][0]
            model.eval()
            with torch.no_grad():
                result = model.sample(test_spectrum[row : row + 1], num_samples=num_samples)
            has_log_prob = isinstance(result, tuple)
            flat_samples = result[0][0] if has_log_prob else result[0]
            log_probs = result[1][0] if has_log_prob else None

            physical = denormalize_design(flat_samples.numpy(), NUM_RES, norm)  # (num_samples, num_res, 5)
            predicted_erp = solve_configs(solver_pool, physical, freq_hz)  # (num_samples, n_freq), real dB
            recon_mse = ((predicted_erp - true_erp[row][None, :]) ** 2).mean(axis=1)

            if has_log_prob:
                log_probs_np = log_probs.numpy()
                confidence = np.exp(log_probs_np - log_probs_np.max())
                confidence = confidence / confidence.sum()
                best_idx = int(log_probs_np.argmax())
                score_label = "log p(design|spectrum)"
                scores = log_probs_np
            else:
                # Diffusion has no tractable density; use spectrum-consistency
                # (negative reconstruction MSE) as an explicit, differently-
                # labeled confidence proxy instead.
                confidence = np.exp(-recon_mse) / np.exp(-recon_mse).sum()
                best_idx = int(recon_mse.argmin())
                score_label = "spectrum-consistency (-MSE, NOT a probability)"
                scores = -recon_mse

            report_lines.append(f"\n--- {name} ({score_label}) ---")
            for i in range(num_samples):
                marker = " <-- best" if i == best_idx else ""
                report_lines.append(
                    f"sample {i + 1}: score={scores[i]:+.4f}  relative_confidence={confidence[i] * 100:5.1f}%"
                    f"  recon_MSE={recon_mse[i]:.4f}{marker}"
                )
                report_lines.append(format_configuration(physical[i]))

            ax = axes[row, col] if num_examples > 1 else axes[col]
            for i in range(num_samples):
                if i == best_idx:
                    continue
                ax.plot(freq_hz, predicted_erp[i], color="#4C72B0", alpha=0.3, lw=1.1)
            ax.plot(freq_hz, predicted_erp[best_idx], color="#C44E52", lw=2.0, label="Best sample")
            ax.plot(freq_hz, true_erp[row], color="black", lw=2, label="Target")
            best_conf = confidence[best_idx] * 100
            ax.text(
                0.02, 0.98,
                f"best: {best_conf:.0f}% rel.\nconfidence",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
            )
            if row == 0:
                ax.set_title(name, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Example {row + 1}\nERP (dB)")
            ax.set_xlabel("Frequency (Hz)")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"Inverse-design validation: {num_samples} sampled designs per method "
        "(best-scoring highlighted red), run through the actual solver (not a neural surrogate)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "validation_reconstructions.png", dpi=150)
    plt.close(fig)
    print(f"Saved validation figure to {out_dir / 'validation_reconstructions.png'}")

    report_path = out_dir / "sample_configurations_and_scores.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"Saved per-sample configurations + scores to {report_path}")

    solver_pool.shutdown()

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, history in histories.items():
        ax.plot(history["val"], label=name, lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss (method-specific NLL/MSE, not comparable across methods)")
    ax.set_title("Inverse model training curves (post-fix: LR schedule + best-checkpoint + KL annealing + cosine diffusion schedule)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {out_dir / 'training_curves.png'}")

    print("\nDONE")


if __name__ == "__main__":
    main()
