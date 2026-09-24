"""Three-step training (Long et al. arXiv:2402.11722, Sec 3.3) for any
model in this package's registry, then forward + inverse evaluation on
this project's config<->ERP problem. Generalized from the original iFNO-
only training script so the same schedule drives iFNO, iDCO and iGNO.

Step 1: train the invertible coupling blocks + P/Q/P'/Q' (every parameter
except the VAE) with J_IFB = J_FWD + J_INV + J_{P,Q'} + J_{P',Q}.
Step 2: train the beta-VAE alone, directly on TRUE designs (no spectrum,
no invertible blocks involved) -- pure design-space pretraining.
Step 3: combine both, continue training every parameter jointly with
J = J_FWD + J_{P,Q'} + J_{P',Q} + J_beta-VAE (now the VAE encodes the
inverse pipeline's own point estimate, not the oracle design).

Evaluation, once trained:
  - Forward: predict_spectrum(configuration) vs the dataset's true ERP on
    the test split -- RMSE/MAE/R^2/Pearson plus per-configuration overlay
    plots, same reporting convention as erp_forward_operators/evaluate_operator.
  - Inverse: sample(spectrum, num_samples) on test spectra, run every
    sampled design through the ACTUAL coupled solver (not a neural
    surrogate -- same principle as erp_inverse_operators/train_all.py's own
    validation), plot best-sample-vs-target overlays.
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from erp_forward_inverse_operators.common import InvertibleOperatorBase
from erp_forward_inverse_operators.registry import INVERTIBLE_OPERATORS
from erp_inverse_operators.common import denormalize_design, prepare_inverse_data
from erp_inverse_operators.evaluate import SPAWN_CONTEXT, solve_configs
from erp_inverse_operators.registry import DESIGN_DIM, NUM_RES
from utils.erp_dataset import denormalize_configuration_array, denormalize_erp_array
from utils.plotting import plot_erp_comparison, plot_prediction_scatter
from utils.support import seed_everything

OUT_DIR = Path("erp_forward_inverse_operators/plots")
MODEL_DIR = Path("erp_forward_inverse_operators/models")

STAGE1_EPOCHS = 6
STAGE2_EPOCHS = 25
STAGE3_EPOCHS = 6
KL_WARMUP_EPOCHS = 8
KL_TARGET_BETA = 0.05
BATCH_SIZE = 128
NUM_CONFIGURATIONS = 100000
DATASET_FILE = ("datasets/dataset_erp_ft_100k_part1.pth", "datasets/dataset_erp_ft_100k_part2.pth")
SEED = 727


def _non_vae_parameters(model: InvertibleOperatorBase):
    return [p for name, p in model.named_parameters() if not name.startswith("vae.")]


def train_stage1(model: InvertibleOperatorBase, loaders, epochs: int, tag: str, lr: float = 5e-4):
    optimizer = torch.optim.Adam(_non_vae_parameters(model), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for spectrum, design in loaders["train"]:
            flat_design = design.reshape(design.shape[0], -1)
            optimizer.zero_grad(set_to_none=True)
            loss = model.stage1_loss(spectrum, flat_design)["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(_non_vae_parameters(model), 5.0)
            optimizer.step()
            total += loss.item() * spectrum.shape[0]
            n += spectrum.shape[0]
        train_loss = total / n
        scheduler.step()

        model.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for spectrum, design in loaders["val"]:
                flat_design = design.reshape(design.shape[0], -1)
                loss = model.stage1_loss(spectrum, flat_design)["total"]
                total += loss.item() * spectrum.shape[0]
                n += spectrum.shape[0]
            val_loss = total / n

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{tag} stage1] epoch {epoch + 1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{tag} stage1] restored best-validation checkpoint (val={best_val:.4f})")
    return history


def train_stage2(model: InvertibleOperatorBase, loaders, epochs: int, tag: str, lr: float = 1e-3):
    optimizer = torch.optim.Adam(model.vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        beta = KL_TARGET_BETA * min(1.0, epoch / KL_WARMUP_EPOCHS)
        model.train()
        total, n = 0.0, 0
        for _spectrum, design in loaders["train"]:
            flat_design = design.reshape(design.shape[0], -1)
            optimizer.zero_grad(set_to_none=True)
            loss = model.stage2_loss(flat_design, beta=beta)["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.vae.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * flat_design.shape[0]
            n += flat_design.shape[0]
        train_loss = total / n
        scheduler.step()

        model.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for _spectrum, design in loaders["val"]:
                flat_design = design.reshape(design.shape[0], -1)
                loss = model.stage2_loss(flat_design, beta=beta)["total"]
                total += loss.item() * flat_design.shape[0]
                n += flat_design.shape[0]
            val_loss = total / n

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.vae.state_dict())
        print(f"[{tag} stage2] epoch {epoch + 1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | beta={beta:.4f}")

    if best_state is not None:
        model.vae.load_state_dict(best_state)
        print(f"[{tag} stage2] restored best-validation VAE checkpoint (val={best_val:.4f})")
    return history


def train_stage3(model: InvertibleOperatorBase, loaders, epochs: int, tag: str, lr: float = 3e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for spectrum, design in loaders["train"]:
            flat_design = design.reshape(design.shape[0], -1)
            optimizer.zero_grad(set_to_none=True)
            loss = model.stage3_loss(spectrum, flat_design, beta=KL_TARGET_BETA)["total"]
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
                flat_design = design.reshape(design.shape[0], -1)
                loss = model.stage3_loss(spectrum, flat_design, beta=KL_TARGET_BETA)["total"]
                total += loss.item() * spectrum.shape[0]
                n += spectrum.shape[0]
            val_loss = total / n

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{tag} stage3] epoch {epoch + 1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{tag} stage3] restored best-validation checkpoint (val={best_val:.4f})")
    return history


def plot_stage_losses(tag: str, histories: dict[str, dict[str, list[float]]], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (stage, history) in zip(axes, histories.items()):
        ax.plot(history["train"], label="train", lw=2)
        ax.plot(history["val"], label="val", lw=2)
        ax.set_title(f"{tag} {stage}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{tag} three-step training (Long et al., arXiv:2402.11722, Sec 3.3)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def evaluate_forward(model: InvertibleOperatorBase, tag: str, dataset, loaders, norm, num_plot: int = 5):
    model.eval()
    true_curves, pred_curves, configs_used = [], [], []
    with torch.no_grad():
        for spectrum, design in loaders["test"]:
            configuration = design
            pred = model.predict_spectrum(configuration).squeeze(-1)  # (B, F) normalized
            true_curves.append(denormalize_erp_array(spectrum, norm))
            pred_curves.append(denormalize_erp_array(pred, norm))
            configs_used.append(design.numpy())
    true = np.concatenate(true_curves, axis=0)
    pred = np.concatenate(pred_curves, axis=0)
    configs_used = np.concatenate(configs_used, axis=0)

    error = pred - true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    tc, pc = true.ravel() - true.mean(), pred.ravel() - pred.mean()
    denom = float(np.sqrt(np.sum(tc**2) * np.sum(pc**2)))
    pearson = float(np.sum(tc * pc) / denom) if denom > 0 else float("nan")
    r2_denom = float(np.sum(tc**2))
    r2 = float(1.0 - np.sum((pred.ravel() - true.ravel()) ** 2) / r2_denom) if r2_denom > 0 else float("nan")

    print("=" * 68)
    print(f"[{tag} forward] test RMSE : {rmse:.4f} dB")
    print(f"[{tag} forward] test MAE  : {mae:.4f} dB")
    print(f"[{tag} forward] Pearson    : {pearson:.4f}")
    print(f"[{tag} forward] R^2        : {r2:.4f}")
    print("=" * 68)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freq_hz = np.asarray(dataset.frequency_values)
    physical_configs = denormalize_configuration_array(configs_used[:num_plot], norm)
    for i in range(min(num_plot, true.shape[0])):
        plot_erp_comparison(
            freq_hz, true[i], pred[i],
            title=f"{tag} (forward) - test configuration {i + 1}",
            configuration=physical_configs[i],
            save_path=OUT_DIR / f"{tag.lower()}_forward_erp_test_config_{i + 1:02d}.png",
            show=False,
        )
    plot_prediction_scatter(
        true, pred, xlabel="Ground Truth ERP (dB)", ylabel=f"Predicted ERP (dB, {tag} forward)",
        title=f"{tag} - predicted vs. ground truth ERP",
        save_path=OUT_DIR / f"{tag.lower()}_forward_prediction_vs_ground_truth.png",
        show=False,
    )
    return {"rmse": rmse, "mae": mae, "pearson": pearson, "r2": r2}


def evaluate_inverse(model: InvertibleOperatorBase, tag: str, dataset, loaders, norm, num_examples: int = 5, num_samples: int = 6):
    model.eval()
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:num_examples]
    freq_hz = np.asarray(dataset.frequency_values)
    true_erp = denormalize_erp_array(test_spectrum, norm)

    with torch.no_grad():
        samples = model.sample(test_spectrum, num_samples=num_samples)  # (num_examples, num_samples, design_dim)
    physical = denormalize_design(samples.numpy(), NUM_RES, norm)  # (num_examples, num_samples, num_res, 5)

    solver_pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT)
    fig, axes = plt.subplots(1, num_examples, figsize=(4.8 * num_examples, 4.2))
    for row in range(num_examples):
        predicted_erp = solve_configs(solver_pool, physical[row], freq_hz)  # (num_samples, n_freq)
        recon_mse = ((predicted_erp - true_erp[row][None, :]) ** 2).mean(axis=1)
        best_idx = int(recon_mse.argmin())

        ax = axes[row] if num_examples > 1 else axes
        for i in range(num_samples):
            if i == best_idx:
                continue
            ax.plot(freq_hz, predicted_erp[i], color="#4C72B0", alpha=0.3, lw=1.1)
        ax.plot(freq_hz, predicted_erp[best_idx], color="#C44E52", lw=2.0, label="Best sample")
        ax.plot(freq_hz, true_erp[row], color="black", lw=2, label="Target")
        ax.set_title(f"Example {row + 1} (best MSE={recon_mse[best_idx]:.3f})", fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        if row == 0:
            ax.set_ylabel("ERP (dB)")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    solver_pool.shutdown()

    fig.suptitle(
        f"{tag} inverse-design validation: {num_samples} posterior samples/target, "
        "best-scoring (vs. actual solver) highlighted red",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{tag.lower()}_inverse_validation_reconstructions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def train_one(
    key: str,
    num_configurations: int = NUM_CONFIGURATIONS,
    batch_size: int = BATCH_SIZE,
    dataset_file=DATASET_FILE,
    seed: int = SEED,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage2_epochs: int = STAGE2_EPOCHS,
    stage3_epochs: int = STAGE3_EPOCHS,
):
    """Train+evaluate exactly one registry entry (e.g. key="2" for iDCO)."""
    spec = INVERTIBLE_OPERATORS[key]
    tag = spec["short"]

    seed_everything(seed)
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=batch_size, dataset_file=dataset_file, seed=seed,
    )
    norm = dataset.norm_params

    model: InvertibleOperatorBase = spec["build"](design_dim=DESIGN_DIM, **spec["model_config"])
    print(f"{tag} params: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "#" * 70 + f"\n{tag} Stage 1: invertible coupling blocks + P/Q/P'/Q'\n" + "#" * 70)
    history1 = train_stage1(model, loaders, epochs=stage1_epochs, tag=tag)

    print("\n" + "#" * 70 + f"\n{tag} Stage 2: beta-VAE pretraining (design space only)\n" + "#" * 70)
    history2 = train_stage2(model, loaders, epochs=stage2_epochs, tag=tag)

    print("\n" + "#" * 70 + f"\n{tag} Stage 3: joint fine-tuning\n" + "#" * 70)
    history3 = train_stage3(model, loaders, epochs=stage3_epochs, tag=tag)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / f"{tag.lower()}.pth"
    torch.save(
        {"model_state_dict": model.state_dict(), "norm_params": dict(norm), "model_config": dict(spec["model_config"])},
        checkpoint_path,
    )
    print(f"Saved {checkpoint_path}")

    plot_stage_losses(
        tag,
        {"stage1 (invertible blocks)": history1, "stage2 (beta-VAE)": history2, "stage3 (joint fine-tune)": history3},
        OUT_DIR / f"{tag.lower()}_training_curves.png",
    )

    forward_metrics = evaluate_forward(model, tag, dataset, loaders, norm)
    evaluate_inverse(model, tag, dataset, loaders, norm)

    print(f"\n{tag} DONE")
    return model, {"stage1": history1, "stage2": history2, "stage3": history3}, forward_metrics


def main(keys: tuple[str, ...] = ("1", "2", "3"), **kwargs):
    results = {}
    for key in keys:
        results[key] = train_one(key, **kwargs)
    return results


if __name__ == "__main__":
    main()
