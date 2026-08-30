"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Neural Operator Main
Description : Interactive entry point for ERP neural operators
==================================================

Run:
    python main.py

Dataset preprocessing is handled by erp_dataset.py, training/evaluation by
neural_operator_utils.py, and all figure creation/saving by plotting.py.
"""

from __future__ import annotations

import gc

import numpy as np
import torch

from utils.erp_dataset import DEFAULT_DATASET_FILE
from operators.operator_registry import OPERATORS
from utils.physics import Lx, Ly, fmin, fmax, k, num_res as default_num_res
from utils.plotting import DEFAULT_PLOTS_DIR, save_operator_experiment_plots


DATASET_FILE = DEFAULT_DATASET_FILE
SEED = 727
PLOTS_DIR = DEFAULT_PLOTS_DIR
PLOT_PIPELINE_VERSION = "main-direct-save-v5"
# Always create the top-level plot folder beside main.py at startup.
# Per-operator subfolders are created automatically when their plots are saved.
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = {
    "1": "train",
    "2": "evaluate",
    "3": "predict",
}


# ==================================================
# Input helpers
# ==================================================


def _prompt_choice(prompt: str, choices: dict[str, object]) -> str:
    while True:
        value = input(prompt).strip()
        if value in choices:
            return value
        print(f"Please choose one of: {', '.join(choices)}")


def _prompt_int(prompt: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return int(default)
        try:
            value = int(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"Please enter an integer >= {minimum}.")


def _prompt_float(prompt: str, default: float, minimum: float = 0.0) -> float:
    while True:
        raw = input(f"{prompt} [{default:g}]: ").strip()
        if not raw:
            return float(default)
        try:
            value = float(raw)
            if value > minimum:
                return value
        except ValueError:
            pass
        print(f"Please enter a number > {minimum}.")


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _prompt_configuration(num_res: int) -> np.ndarray:
    """Read one raw configuration in [f_t, x, y] order."""
    configuration = np.empty((num_res, 3), dtype=np.float32)

    print("\nEnter resonator configuration in [f_t, x, y] form.")
    print(f"Allowed tuning-frequency range: {fmin:g} to {fmax:g} Hz")
    print(f"Plate range: 0 <= x <= {Lx:g} m, 0 <= y <= {Ly:g} m")

    for i in range(num_res):
        while True:
            try:
                f_t = float(input(f"\nResonator {i + 1} tuning frequency f_t (Hz): ").strip())
                x = float(input(f"Resonator {i + 1} x position (m): ").strip())
                y = float(input(f"Resonator {i + 1} y position (m): ").strip())
            except ValueError:
                print("Please enter numeric values.")
                continue

            if not (fmin <= f_t <= fmax):
                print(f"f_t must be between {fmin:g} and {fmax:g} Hz.")
                continue
            if not (0.0 <= x <= Lx and 0.0 <= y <= Ly):
                print(f"Position must satisfy 0 <= x <= {Lx:g}, 0 <= y <= {Ly:g}.")
                continue

            configuration[i] = [f_t, x, y]
            mass = k / (2.0 * np.pi * f_t) ** 2
            print(f"Derived solver mass: {mass:.6f} kg")
            break

    print("\nConfiguration used:")
    print(configuration)
    return configuration


# ==================================================
# Menus / reporting
# ==================================================


def _print_operator_menu() -> None:
    print("\nAvailable neural operators")
    print("=" * 52)
    for key, spec in OPERATORS.items():
        print(f"{key}. {spec['name']} ({spec['short']})")
    print("9. Train and evaluate ALL models")
    print("=" * 52)


def _print_action_menu() -> None:
    print("\nOperation")
    print("1. Train")
    print("2. Evaluate")
    print("3. Predict")


def _print_comparison_table(rows: list[dict[str, object]]) -> None:
    """Print a dependency-free final comparison table for all operators."""
    columns = [
        ("Model", "model", None),
        ("Final Train Loss", "final_train_loss", ".6e"),
        ("Final Val Loss", "final_val_loss", ".6e"),
        ("Best Val Loss", "best_val_loss", ".6e"),
        ("Test MSE", "mse", ".6f"),
        ("Test RMSE (dB)", "rmse", ".6f"),
        ("Test MAE (dB)", "mae", ".6f"),
        ("Peak Freq MAE (Hz)", "peak_frequency_mae_hz", ".4f"),
        ("Peak Amp MAE (dB)", "peak_amplitude_mae_db", ".6f"),
    ]

    formatted_rows: list[list[str]] = []
    for row in rows:
        formatted = []
        for _, key, fmt in columns:
            value = row[key]
            formatted.append(str(value) if fmt is None else format(float(value), fmt))
        formatted_rows.append(formatted)

    widths = [
        max(len(header), *(len(row[idx]) for row in formatted_rows))
        for idx, (header, _, _) in enumerate(columns)
    ]
    header_line = " | ".join(
        header.ljust(widths[idx]) for idx, (header, _, _) in enumerate(columns)
    )
    separator = "-+-".join("-" * width for width in widths)

    print("\n" + "=" * len(header_line))
    print("FINAL ALL-MODEL COMPARISON")
    print("=" * len(header_line))
    print(header_line)
    print(separator)
    for row in formatted_rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))
    print("=" * len(header_line))


# ==================================================
# All-model workflow
# ==================================================


def train_all_models(
    *,
    num_configurations: int = 500,
    batch_size: int = 16,
    dataset_file: str = DATASET_FILE,
    regenerate_dataset: bool = False,
    seed: int = SEED,
) -> dict[str, object]:
    """Train and evaluate every registered operator sequentially.

    This workflow is deliberately non-interactive for Matplotlib: every plot is
    saved to ``plots/<operator>/`` and immediately closed. No ``plt.show()`` is
    called, so training proceeds continuously without waiting for plot windows.
    """
    comparison_rows: list[dict[str, object]] = []
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("TRAIN + EVALUATE ALL ERP NEURAL OPERATORS")
    print(f"Dataset        : {dataset_file}")
    print(f"Configurations : {num_configurations}")
    print(f"Batch size     : {batch_size}")
    print(f"Seed           : {seed}")
    print(f"Regenerate     : {regenerate_dataset}")
    print(f"Plots folder   : {PLOTS_DIR.resolve()}")
    print(f"Plot pipeline  : {PLOT_PIPELINE_VERSION}")
    print("Interactive plots: disabled for uninterrupted batch training")
    print("=" * 76)

    for model_index, spec in enumerate(OPERATORS.values()):
        epochs = int(spec["epochs"])
        learning_rate = float(spec["lr"])
        regenerate_this_model = bool(regenerate_dataset and model_index == 0)

        print("\n" + "#" * 76)
        print(f"Training {spec['name']} ({spec['short']})")
        print(f"epochs={epochs} | lr={learning_rate:g}")
        print("#" * 76)

        result = spec["runner"](
            action="train",
            num_configurations=num_configurations,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dataset_file=dataset_file,
            regenerate_dataset=regenerate_this_model,
            seed=seed,
            plot=False,
            save_plots=False,           # main.py saves explicitly below
            plots_dir=PLOTS_DIR,
            num_evaluation_plots=5,
            evaluate_after_training=True,
        )

        history = result["history"]
        metrics = result["metrics"]

        # Explicit main -> plotting.py call.  This is intentionally outside the
        # training utility so plot saving cannot be skipped by an internal flag.
        saved_plot_dir = save_operator_experiment_plots(
            spec["short"],
            history=history,
            metrics=metrics,
            frequency_values=result["dataset"].frequency_values,
            plots_dir=PLOTS_DIR,
            num_configurations=5,
            show=False,
        )
        expected_plot_count = 2 + min(5, int(np.asarray(metrics["targets"]).shape[0]))
        actual_plot_count = sum(
            1 for path in saved_plot_dir.glob("*.png") if path.stat().st_size > 0
        )
        if actual_plot_count < expected_plot_count:
            raise RuntimeError(
                f"{spec['short']} plot verification failed: expected at least "
                f"{expected_plot_count} PNG files in {saved_plot_dir}, "
                f"found {actual_plot_count}."
            )
        print(
            f"{spec['short']} plot verification passed: "
            f"{actual_plot_count} PNG file(s) in {saved_plot_dir.resolve()}"
        )
        comparison_rows.append(
            {
                "model": spec["short"],
                "final_train_loss": history["train"][-1],
                "final_val_loss": history["val"][-1],
                "best_val_loss": min(history["val"]),
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "peak_frequency_mae_hz": metrics["peak_frequency_mae_hz"],
                "peak_amplitude_mae_db": metrics["peak_amplitude_mae_db"],
            }
        )

        # Release the complete experiment result before the next architecture.
        del history, metrics, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_comparison_table(comparison_rows)
    return {
        "action": "train_all",
        "comparison": comparison_rows,
        "plots_dir": str(PLOTS_DIR),
    }


def main_all_models():
    """Interactive input collection for the non-blocking all-model workflow."""
    print("\nAll-model training parameters")
    print("Per-model defaults come from each operator's DEFAULT_TRAINING_CONFIG.")
    for spec in OPERATORS.values():
        print(
            f"  {spec['short']:>5s}: epochs={int(spec['epochs'])}, "
            f"lr={float(spec['lr']):g}"
        )

    num_configurations = _prompt_int(
        "Number of configurations to use for every model",
        default=5000,
        minimum=3,
    )
    batch_size = _prompt_int(
        "Batch size (complete ERP spectra per batch)",
        default=64,
        minimum=1,
    )
    regenerate_dataset = _prompt_yes_no(
        "Regenerate and overwrite the ERP dataset before training",
        default=False,
    )

    print(f"All figures will be saved under: {PLOTS_DIR}")
    print("No figures will be displayed during all-model training.")

    return train_all_models(
        num_configurations=num_configurations,
        batch_size=batch_size,
        dataset_file=DATASET_FILE,
        regenerate_dataset=regenerate_dataset,
        seed=SEED,
    )


# ==================================================
# Single-operator workflow
# ==================================================


def main():
    """Interactive entry point for all ERP neural operators."""
    _print_operator_menu()
    operator_choices = {**OPERATORS, "9": None}
    operator_key = _prompt_choice("Select operator/workflow: ", operator_choices)

    if operator_key == "9":
        return main_all_models()

    spec = OPERATORS[operator_key]
    _print_action_menu()
    action = ACTIONS[_prompt_choice("Select operation: ", ACTIONS)]
    runner = spec["runner"]

    print("\n" + "=" * 68)
    print(f"Operator : {spec['name']} ({spec['short']})")
    print(f"Action   : {action}")
    print(f"Dataset  : {DATASET_FILE}")
    print(f"Seed     : {SEED}")
    print(f"Plots    : {PLOTS_DIR}")
    print("=" * 68)

    if action == "train":
        num_configurations = _prompt_int(
            "Number of configurations to use", default=5000, minimum=3
        )
        batch_size = _prompt_int(
            "Batch size (complete ERP spectra per batch)", default=16, minimum=1
        )
        epochs = _prompt_int(
            "Number of epochs", default=int(spec["epochs"]), minimum=1
        )
        learning_rate = _prompt_float(
            "Learning rate", default=float(spec["lr"]), minimum=0.0
        )
        regenerate_dataset = _prompt_yes_no(
            "Regenerate and overwrite the ERP dataset before training", default=False
        )
        show_plot = _prompt_yes_no(
            "Also display the saved training and evaluation plots", default=False
        )

        # Train one operator and evaluate it immediately afterwards so a
        # standalone model run saves the same diagnostic plots as the
        # all-model workflow. Plot creation is kept in main.py explicitly.
        result = runner(
            action="train",
            num_configurations=num_configurations,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            dataset_file=DATASET_FILE,
            regenerate_dataset=regenerate_dataset,
            seed=SEED,
            plot=False,
            save_plots=False,
            plots_dir=PLOTS_DIR,
            num_evaluation_plots=5,
            evaluate_after_training=True,
        )

        saved_plot_dir = save_operator_experiment_plots(
            spec["short"],
            history=result["history"],
            metrics=result["metrics"],
            frequency_values=result["dataset"].frequency_values,
            plots_dir=PLOTS_DIR,
            num_configurations=5,
            show=show_plot,
        )
        print(f"Saved {spec['short']} training/evaluation plots to: {saved_plot_dir.resolve()}")
        return result

    if action == "evaluate":
        batch_size = _prompt_int("Evaluation batch size", default=16, minimum=1)
        show_plot = _prompt_yes_no(
            "Also display the saved test ERP and parity plots", default=False
        )

        result = runner(
            action="evaluate",
            batch_size=batch_size,
            dataset_file=DATASET_FILE,
            seed=SEED,
            plot=False,
            save_plots=False,
            plots_dir=PLOTS_DIR,
            num_evaluation_plots=5,
        )
        saved_plot_dir = save_operator_experiment_plots(
            spec["short"],
            metrics=result["metrics"],
            frequency_values=result["frequency_values"],
            plots_dir=PLOTS_DIR,
            num_configurations=5,
            show=show_plot,
        )
        print(f"Saved {spec['short']} evaluation plots to: {saved_plot_dir.resolve()}")
        return result

    print("\nPrediction configuration")
    print("1. Enter a new configuration manually")
    print("2. Use the first configuration from the saved test split")
    prediction_choice = _prompt_choice("Select prediction input: ", {"1": None, "2": None})
    configuration = (
        _prompt_configuration(default_num_res) if prediction_choice == "1" else None
    )
    show_plot = _prompt_yes_no(
        "Also display the saved solver vs neural-operator ERP spectrum", default=False
    )

    result = runner(
        action="predict",
        dataset_file=DATASET_FILE,
        seed=SEED,
        configuration=configuration,
        plot=False,
        save_plots=False,
        plots_dir=PLOTS_DIR,
    )
    saved_plot_dir = save_operator_experiment_plots(
        spec["short"],
        prediction=result["prediction"],
        plots_dir=PLOTS_DIR,
        show=show_plot,
    )
    print(f"Saved {spec['short']} prediction plot to: {saved_plot_dir.resolve()}")
    return result


if __name__ == "__main__":
    results = main()
