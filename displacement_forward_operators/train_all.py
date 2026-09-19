"""Train, evaluate, and plot one displacement forward operator end to end.

Usage: python3 -m displacement_forward_operators.train_all --key 1
(key matches operator_registry.OPERATORS, "1".."10")

Saves:
  displacement_forward_operators/models/<SHORT>.pth       -- checkpoint
  displacement_forward_operators/plots/<SHORT>/loss_curve.png
  displacement_forward_operators/plots/<SHORT>/erp_spectrum_test_config_0N.png (x5)
  displacement_forward_operators/plots/<SHORT>/prediction_vs_ground_truth.png

Scope note (CPU-only environment): uses all 200 configurations (for
train/val/test diversity) but a 100-of-480 collocation-point subsample per
configuration to keep epoch time tractable -- the physics loss's nested
4th-order autograd is expensive per step regardless of batch size, and
full 480-point density across 10 architectures would take many hours
each. This is a real, non-trivial training run, not a smoke test, but
it's scoped for CPU feasibility, not for a maximal/production result.
"""

from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from displacement_forward_operators.displacement_operator_utils import (
    build_displacement_loaders,
    evaluate_displacement_operator,
    train_displacement_operator,
)
from displacement_forward_operators.operator_registry import OPERATORS
from displacement_forward_operators.physics_loss import make_physics_loss_fn
from forward_operators.neural_operator_utils import parameter_count
from utils.field_dataset import load_field_dataset_shards
from utils.support import seed_everything

POINTS_PER_CONFIG_USED = 100  # of 480 available, subsampled for CPU feasibility
EPOCHS = 10
BATCH_SIZE = 128
PHYSICS_WEIGHT = 0.02
PHYSICS_POINTS_PER_STEP = 16
SEED = 727


def load_scoped_dataset():
    files = [f"datasets/dataset_field_displacement_part{i}.pth" for i in range(1, 5)]
    raw = load_field_dataset_shards(files)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(raw["collocation_points"].shape[1], size=POINTS_PER_CONFIG_USED, replace=False)
    return {
        "configuration_features": raw["configuration_features"],
        "frequency_values": raw["frequency_values"],
        "collocation_points": raw["collocation_points"][:, idx, :],
        "real_displacement": raw["real_displacement"][:, idx, :],
        "imag_displacement": raw["imag_displacement"][:, idx, :],
        "num_res": raw["num_res"],
    }


def run_one(key: str):
    seed_everything(SEED)
    spec = OPERATORS[key]
    name = spec["short"]
    print(f"\n{'#' * 70}\nTraining {name} (displacement field operator)\n{'#' * 70}")

    dataset_dict = load_scoped_dataset()
    data = build_displacement_loaders(dataset_dict, batch_size=BATCH_SIZE, seed=SEED)
    print(f"train={len(data['loaders']['train'].dataset)} val={len(data['loaders']['val'].dataset)} "
          f"test={len(data['loaders']['test'].dataset)} examples")

    model = spec["build_model"](num_res=data["num_res"], **spec["model_config"])
    print(f"{name} params: {parameter_count(model):,}")

    physics_fn = make_physics_loss_fn(
        data["norm_params"], num_configs=64, points_per_step=PHYSICS_POINTS_PER_STEP, seed=SEED
    )

    t0 = time.time()
    model, history = train_displacement_operator(
        model, data["loaders"], epochs=EPOCHS, lr=spec["lr"],
        physics_loss_fn=physics_fn, physics_weight=PHYSICS_WEIGHT, operator_name=name,
    )
    train_time = time.time() - t0
    print(f"{name} training time: {train_time:.1f}s")

    plot_dir = f"displacement_forward_operators/plots/{name}"
    import os
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["train"], label="train", lw=2)
    axes[0].plot(history["val"], label="val", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Data loss (MSE, normalized displacement)")
    axes[0].set_title(f"{name}: data loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history["physics"], color="#C44E52", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Physics residual loss (normalized)")
    axes[1].set_title(f"{name}: physics loss")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"{name} training curves ({EPOCHS} epochs, {POINTS_PER_CONFIG_USED} pts/config)")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {plot_dir}/loss_curve.png")

    result = evaluate_displacement_operator(
        model, data, num_plot=5, grid_nx=40, grid_ny=15, operator_name=name, plot=False,
    )
    print(f"{name} field-reconstructed ERP: RMSE={result['rmse']:.3f} dB MAE={result['mae']:.3f} dB "
          f"R^2={result['r2_score']:.3f} Pearson={result['pearson_correlation']:.3f}")

    checkpoint_path = f"displacement_forward_operators/models/{name.lower()}.pth"
    torch.save(
        {"state_dict": model.state_dict(), "model_config": spec["model_config"],
         "norm_params": data["norm_params"], "num_res": data["num_res"],
         "history": history, "eval_rmse": result["rmse"], "eval_mae": result["mae"],
         "eval_r2": result["r2_score"], "eval_pearson": result["pearson_correlation"]},
        checkpoint_path,
    )
    print(f"Saved {checkpoint_path}")
    return name, result, train_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    run_one(args.key)
