"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Learning-Rate Sweep Main
Description : Run one ERP neural operator with several learning rates and
              report/save the best learning rate by validation loss.
==================================================

Run:
    python long_main.py

This file reuses the same operator registry, dataset, training, evaluation,
seed, and plotting conventions as main.py. Each trial uses the same seed and
configuration subset/split so the learning-rate comparison is fair.

The winning learning rate is selected using the minimum validation loss seen
throughout training, NOT test loss. Test metrics are reported only as extra
information.
"""

from __future__ import annotations

import gc
import math
import shutil
from pathlib import Path

import torch

from utils.cli import (
    DATASET_FILE,
    PLOTS_DIR,
    SEED,
    _print_operator_menu,
    _prompt_choice,
    _prompt_int,
    _prompt_yes_no,
)
from forward_operators.operator_registry import OPERATORS
from utils.plotting import save_operator_experiment_plots


# ==================================================
# Sweep defaults / helpers
# ==================================================

SWEEP_MODELS_DIR = Path("models") / "lr_sweeps"
SWEEP_PLOTS_DIR = Path(PLOTS_DIR) / "lr_sweeps"


def _default_learning_rates(base_lr: float) -> list[float]:
    """Return five rates centered on the operator's registry default."""
    base_lr = float(base_lr)
    return [
        base_lr / 4.0,
        base_lr / 2.0,
        base_lr,
        base_lr * 2.0,
        base_lr * 4.0,
    ]


def _format_lr_for_filename(lr: float) -> str:
    """Create a compact filesystem-safe learning-rate string."""
    return f"{float(lr):.3e}".replace("+", "").replace("-", "m").replace(".", "p")


def _prompt_learning_rates(default_rates: list[float]) -> list[float]:
    """Read a comma-separated list of positive learning rates."""
    default_text = ", ".join(f"{lr:g}" for lr in default_rates)

    while True:
        raw = input(
            "Learning rates to test, comma-separated "
            f"[{default_text}]: "
        ).strip()

        if not raw:
            return list(default_rates)

        try:
            rates = [float(value.strip()) for value in raw.split(",")]
        except ValueError:
            print("Please enter valid numbers separated by commas.")
            continue

        if not rates or any((not math.isfinite(lr) or lr <= 0.0) for lr in rates):
            print("Every learning rate must be a finite number > 0.")
            continue

        # Preserve user order while removing exact duplicates.
        unique_rates: list[float] = []
        for lr in rates:
            if lr not in unique_rates:
                unique_rates.append(lr)
        return unique_rates


def _default_checkpoint_path(operator_name: str) -> Path:
    """Match run_operator_experiment's default checkpoint naming."""
    return Path("models") / f"{operator_name.lower()}_erp.pth"


def _copy_trial_checkpoint(
    operator_name: str,
    short_name: str,
    learning_rate: float,
) -> Path:
    """Copy the just-trained default checkpoint to a unique sweep checkpoint."""
    source = _default_checkpoint_path(operator_name)
    if not source.exists():
        raise FileNotFoundError(
            f"Expected checkpoint {source} was not created by the training run."
        )

    SWEEP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = SWEEP_MODELS_DIR / (
        f"{short_name.lower()}_lr_{_format_lr_for_filename(learning_rate)}.pth"
    )
    shutil.copy2(source, destination)
    return destination


def _restore_best_checkpoint(operator_name: str, best_checkpoint: Path) -> Path:
    """Put the winning trial back at the normal checkpoint path."""
    destination = _default_checkpoint_path(operator_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_checkpoint, destination)
    return destination


# ==================================================
# Reporting
# ==================================================


def _print_lr_table(rows: list[dict[str, object]]) -> None:
    columns = [
        ("Learning Rate", "learning_rate", ".6g"),
        ("Best Val Loss", "best_val_loss", ".6e"),
        ("Final Train Loss", "final_train_loss", ".6e"),
        ("Final Val Loss", "final_val_loss", ".6e"),
        ("Test RMSE (dB)", "rmse", ".6f"),
        ("Test MAE (dB)", "mae", ".6f"),
        ("Peak Freq MAE (Hz)", "peak_frequency_mae_hz", ".4f"),
    ]

    formatted_rows: list[list[str]] = [
        [format(float(row[key]), fmt) for _, key, fmt in columns] for row in rows
    ]

    widths = [
        max(len(header), *(len(row[idx]) for row in formatted_rows))
        for idx, (header, _, _) in enumerate(columns)
    ]
    header_line = " | ".join(
        header.ljust(widths[idx]) for idx, (header, _, _) in enumerate(columns)
    )
    separator = "-+-".join("-" * width for width in widths)

    print("\n" + "=" * len(header_line))
    print("LEARNING-RATE SWEEP RESULTS")
    print("=" * len(header_line))
    print(header_line)
    print(separator)
    for row in formatted_rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    print("=" * len(header_line))


# ==================================================
# Learning-rate sweep
# ==================================================


def run_learning_rate_sweep(
    *,
    operator_key: str,
    learning_rates: list[float],
    num_configurations: int = 500,
    batch_size: int = 16,
    epochs: int | None = None,
    dataset_file: str = DATASET_FILE,
    regenerate_dataset: bool = False,
    seed: int = SEED,
    save_trial_plots: bool = True,
) -> dict[str, object]:
    """Train one registered operator at several initial learning rates.

    Selection criterion
    -------------------
    The best learning rate is the one with the smallest value of
    ``min(history["val"])``. This keeps the test split out of hyperparameter
    selection.

    Notes
    -----
    ``train_operator`` uses CosineAnnealingLR, so the values tested here are the
    *initial* AdamW learning rates. The scheduler then anneals each rate during
    its training run.
    """
    if operator_key not in OPERATORS:
        raise ValueError(f"Unknown operator key: {operator_key}")
    if not learning_rates:
        raise ValueError("learning_rates must contain at least one value.")
    if any((not math.isfinite(float(lr)) or float(lr) <= 0.0) for lr in learning_rates):
        raise ValueError("Every learning rate must be finite and > 0.")

    spec = OPERATORS[operator_key]
    runner = spec["runner"]
    short_name = str(spec["short"])
    epochs = int(spec["epochs"] if epochs is None else epochs)

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_checkpoint: Path | None = None
    operator_name: str | None = None

    print("\n" + "=" * 76)
    print(f"LEARNING-RATE SWEEP: {spec['name']} ({short_name})")
    print(f"Dataset        : {dataset_file}")
    print(f"Configurations : {num_configurations}")
    print(f"Batch size     : {batch_size}")
    print(f"Epochs/trial   : {epochs}")
    print(f"Seed           : {seed}")
    print("Learning rates : " + ", ".join(f"{float(lr):g}" for lr in learning_rates))
    print("Winner metric  : minimum validation loss")
    print("=" * 76)

    for trial_index, learning_rate in enumerate(learning_rates, start=1):
        learning_rate = float(learning_rate)
        # Only the first trial may regenerate the raw dataset. Every later trial
        # reuses that same file and the same seed-driven subset/split.
        regenerate_this_trial = bool(regenerate_dataset and trial_index == 1)

        print("\n" + "#" * 76)
        print(
            f"Trial {trial_index}/{len(learning_rates)} | "
            f"{short_name} | initial lr={learning_rate:g}"
        )
        print("#" * 76)

        result = runner(
            action="train",
            num_configurations=num_configurations,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dataset_file=dataset_file,
            regenerate_dataset=regenerate_this_trial,
            seed=seed,
            plot=False,
            save_plots=False,
            plots_dir=PLOTS_DIR,
            num_evaluation_plots=5,
            evaluate_after_training=True,
        )

        history = result["history"]
        metrics = result["metrics"]
        operator_name = str(result["operator_name"])

        trial_checkpoint = _copy_trial_checkpoint(
            operator_name,
            short_name,
            learning_rate,
        )

        if save_trial_plots:
            trial_plot_dir = SWEEP_PLOTS_DIR / (
                f"{short_name}_lr_{_format_lr_for_filename(learning_rate)}"
            )
            save_operator_experiment_plots(
                short_name,
                history=history,
                metrics=metrics,
                frequency_values=result["dataset"].frequency_values,
                plots_dir=trial_plot_dir,
                num_configurations=5,
                show=False,
            )

        row = {
            "learning_rate": learning_rate,
            "best_val_loss": float(min(history["val"])),
            "final_train_loss": float(history["train"][-1]),
            "final_val_loss": float(history["val"][-1]),
            "mse": float(metrics["mse"]),
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "peak_frequency_mae_hz": float(metrics["peak_frequency_mae_hz"]),
            "peak_amplitude_mae_db": float(metrics["peak_amplitude_mae_db"]),
            "checkpoint": str(trial_checkpoint),
        }
        rows.append(row)

        if best_row is None or row["best_val_loss"] < best_row["best_val_loss"]:
            best_row = row
            best_checkpoint = trial_checkpoint

        # Release the complete experiment before the next learning rate.
        del history, metrics, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert best_row is not None
    assert best_checkpoint is not None
    assert operator_name is not None

    # Sort only for reporting. Selection above is independent of table order.
    rows_sorted = sorted(rows, key=lambda row: float(row["best_val_loss"]))
    _print_lr_table(rows_sorted)

    restored_checkpoint = _restore_best_checkpoint(operator_name, best_checkpoint)

    print("\n" + "=" * 76)
    print("BEST LEARNING RATE")
    print("=" * 76)
    print(f"Operator       : {spec['name']} ({short_name})")
    print(f"Learning rate  : {float(best_row['learning_rate']):g}")
    print(f"Best val loss  : {float(best_row['best_val_loss']):.6e}")
    print(f"Test RMSE      : {float(best_row['rmse']):.6f} dB")
    print(f"Test MAE       : {float(best_row['mae']):.6f} dB")
    print(f"Best checkpoint: {restored_checkpoint}")
    print("=" * 76)

    return {
        "action": "learning_rate_sweep",
        "operator": short_name,
        "operator_name": operator_name,
        "results": rows_sorted,
        "best_learning_rate": float(best_row["learning_rate"]),
        "best_validation_loss": float(best_row["best_val_loss"]),
        "best_checkpoint": str(restored_checkpoint),
    }


# ==================================================
# Interactive entry point
# ==================================================


def main() -> dict[str, object]:
    """Interactive learning-rate sweep using the same operators as main.py."""
    _print_operator_menu()
    # long_main.py sweeps one architecture at a time; "10"/"11" from main.py
    # are not included because one global LR across different architectures
    # is not a meaningful single hyperparameter comparison.
    operator_key = _prompt_choice(
        "Select operator to tune (1-9): ",
        OPERATORS,
    )
    spec = OPERATORS[operator_key]

    print("\nLearning-rate sweep parameters")
    num_configurations = _prompt_int(
        "Number of configurations to use",
        default=500,
        minimum=3,
    )
    batch_size = _prompt_int(
        "Batch size (complete ERP spectra per batch)",
        default=16,
        minimum=1,
    )
    epochs = _prompt_int(
        "Epochs per learning-rate trial",
        default=int(spec["epochs"]),
        minimum=1,
    )

    default_rates = _default_learning_rates(float(spec["lr"]))
    learning_rates = _prompt_learning_rates(default_rates)

    regenerate_dataset = _prompt_yes_no(
        "Regenerate and overwrite the ERP dataset before the first trial",
        default=False,
    )
    save_trial_plots = _prompt_yes_no(
        "Save loss/evaluation plots for every learning-rate trial",
        default=True,
    )

    return run_learning_rate_sweep(
        operator_key=operator_key,
        learning_rates=learning_rates,
        num_configurations=num_configurations,
        batch_size=batch_size,
        epochs=epochs,
        dataset_file=DATASET_FILE,
        regenerate_dataset=regenerate_dataset,
        seed=SEED,
        save_trial_plots=save_trial_plots,
    )


if __name__ == "__main__":
    results = main()
