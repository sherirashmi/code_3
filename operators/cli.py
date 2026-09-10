"""Interactive CLI for forward and inverse ERP neural operators.

``main.py`` at the repo root is a thin entry point; every prompt, menu, and
orchestration function lives here. Two top-level workflows:

- **Forward** (configuration -> ERP spectrum): any single operator from
  ``operators.operator_registry.OPERATORS``, or all of them at once.
- **Inverse** (ERP spectrum -> resonator configuration): any single model
  from ``inverse_operators.registry.INVERSE_MODELS``, or all of them at
  once (which also offers the solver-scored aggregate evaluations built in
  ``inverse_operators/evaluate.py`` and ``evaluate_design.py``).

Single inverse models only support "train" here, not "evaluate" --
``evaluate.py``/``evaluate_design.py`` are whole-cohort comparisons by
design (every candidate design is scored via the actual physics solver
against the *other* models' candidates too), not a per-model action, so
they only appear under "train ALL inverse models" below.
"""

from __future__ import annotations

import gc

import numpy as np
import torch

from utils.erp_dataset import DEFAULT_DATASET_FILE
from operators.operator_registry import OPERATORS
from utils.physics import Lx, Ly, fmin, fmax, m_min, m_max, num_res as default_num_res
from utils.plotting import (
    DEFAULT_PLOTS_DIR,
    save_all_model_comparison_plots,
    save_operator_experiment_plots,
)

from inverse_operators.registry import INVERSE_MODELS
from inverse_operators.train_all import train_one_inverse_model
from inverse_operators import evaluate as inverse_evaluate
from inverse_operators import evaluate_design as inverse_evaluate_design


DATASET_FILE = DEFAULT_DATASET_FILE
INVERSE_DATASET_FILE = DEFAULT_DATASET_FILE

# Stopgap for checkpoints saved before ERPDataset.preprocessing_state()
# started recording which raw file its selected_source_ids came from (see
# operators/neural_operator_utils.py's run_operator_experiment): models/
# {dno,dco,gno}_erp.pth on disk right now were trained on the 100k sharded
# dataset, not the 10k DATASET_FILE every other operator's checkpoint uses,
# and have no self-describing metadata to fall back on. Evaluate/predict for
# these 3 keys need the matching file until they're retrained again (which
# would stamp the new field and make this override redundant, though still
# harmless to leave in place).
_HUNDRED_K_DATASET_FILES = [
    "datasets/dataset_erp_ft_100k_part1.pth",
    "datasets/dataset_erp_ft_100k_part2.pth",
]
DATASET_FILE_OVERRIDES: dict[str, object] = {
    "2": _HUNDRED_K_DATASET_FILES,  # DNO
    "4": _HUNDRED_K_DATASET_FILES,  # DCO
    "5": _HUNDRED_K_DATASET_FILES,  # GNO
}
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


def _prompt_optional_int(prompt: str, minimum: int = 1) -> int | None:
    """Blank means "use the model's own default"."""
    while True:
        raw = input(f"{prompt} [blank = use each model's own default]: ").strip()
        if not raw:
            return None
        try:
            value = int(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"Please enter an integer >= {minimum}, or leave blank.")


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


def _prompt_lbfgs_epochs() -> int:
    """Ask whether to run an L-BFGS full-batch fine-tuning phase after AdamW."""
    if not _prompt_yes_no(
        "Run an L-BFGS fine-tuning phase after AdamW training", default=False
    ):
        return 0
    return _prompt_int(
        "Number of L-BFGS steps (each does an internal strong-Wolfe line search "
        "over the full training set)",
        default=20,
        minimum=1,
    )


def _parse_operator_selection(raw: str) -> list[dict[str, object]] | None:
    """Parse a comma/space-separated operator-number string, e.g. "2,4,5".

    Blank input means "all operators" (returns None, so callers can fall
    back to the full registry). Order and duplicates in ``raw`` are
    preserved as given (minus exact repeats) so "5,2,5" trains STO then
    DON, not DON then STO twice. Returns None on any invalid token so the
    caller can re-prompt; never raises.
    """
    raw = raw.strip()
    if not raw:
        return None
    tokens = [t for t in raw.replace(",", " ").split() if t]
    invalid = [t for t in tokens if t not in OPERATORS]
    if invalid:
        return None
    seen: set[str] = set()
    ordered_keys = [t for t in tokens if not (t in seen or seen.add(t))]
    return [OPERATORS[key] for key in ordered_keys]


def _prompt_operator_selection(prompt: str) -> list[dict[str, object]]:
    """Prompt for which operators to include; blank selects every operator."""
    menu = ", ".join(f"{key}={spec['short']}" for key, spec in OPERATORS.items())
    while True:
        raw = input(f"{prompt} [blank = all -- {menu}]: ")
        selected = _parse_operator_selection(raw)
        if raw.strip() and selected is None:
            valid = ", ".join(OPERATORS)
            print(f"Unrecognized operator number(s) in '{raw.strip()}'. Valid keys: {valid}")
            continue
        return selected if selected is not None else list(OPERATORS.values())


def _parse_inverse_selection(raw: str) -> list[str] | None:
    """Same as ``_parse_operator_selection`` but for INVERSE_MODELS keys."""
    raw = raw.strip()
    if not raw:
        return None
    tokens = [t for t in raw.replace(",", " ").split() if t]
    invalid = [t for t in tokens if t not in INVERSE_MODELS]
    if invalid:
        return None
    seen: set[str] = set()
    return [t for t in tokens if not (t in seen or seen.add(t))]


def _prompt_inverse_selection(prompt: str) -> list[str]:
    menu = ", ".join(f"{key}={spec['short']}" for key, spec in INVERSE_MODELS.items())
    while True:
        raw = input(f"{prompt} [blank = all -- {menu}]: ")
        selected = _parse_inverse_selection(raw)
        if raw.strip() and selected is None:
            valid = ", ".join(INVERSE_MODELS)
            print(f"Unrecognized model number(s) in '{raw.strip()}'. Valid keys: {valid}")
            continue
        return selected if selected is not None else list(INVERSE_MODELS.keys())


def _prompt_configuration(num_res: int) -> np.ndarray:
    """Read one raw configuration in [m, k, f_t, x, y] order.

    ``m`` and ``f_t`` are the two quantities actually entered (mass and
    tuning frequency); stiffness ``k = m*(2*pi*f_t)**2`` is derived, matching
    how the dataset itself is generated (see utils/erp_dataset.py).
    """
    configuration = np.empty((num_res, 5), dtype=np.float32)

    print("\nEnter resonator configuration as mass, tuning frequency, and position.")
    print(f"Practical mass range: {m_min:g} to {m_max:g} kg")
    print(f"Allowed tuning-frequency range: {fmin:g} to {fmax:g} Hz")
    print(f"Plate range: 0 <= x <= {Lx:g} m, 0 <= y <= {Ly:g} m")

    for i in range(num_res):
        while True:
            try:
                m = float(input(f"\nResonator {i + 1} mass m (kg): ").strip())
                f_t = float(input(f"Resonator {i + 1} tuning frequency f_t (Hz): ").strip())
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

            k_val = m * (2.0 * np.pi * f_t) ** 2
            configuration[i] = [m, k_val, f_t, x, y]
            print(f"Derived stiffness: {k_val:.3f} N/m")
            break

    print("\nConfiguration used ([m, k, f_t, x, y] per resonator):")
    print(configuration)
    return configuration


# ==================================================
# Menus / reporting
# ==================================================


def _print_operator_menu() -> None:
    print("\nAvailable forward neural operators")
    print("=" * 52)
    for key, spec in OPERATORS.items():
        print(f"{key}. {spec['name']} ({spec['short']})")
    print(f"{len(OPERATORS) + 1}. Train and evaluate ALL forward operators")
    print("=" * 52)


def _print_inverse_menu() -> None:
    print("\nAvailable inverse (ERP -> design) models")
    print("=" * 52)
    for key, spec in INVERSE_MODELS.items():
        print(f"{key}. {spec['name']} ({spec['short']})")
    print(f"{len(INVERSE_MODELS) + 1}. Train ALL inverse models "
          "(+ optional solver-scored evaluation)")
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
        ("Pearson r", "pearson_global", ".6f"),
        ("Mean Spectrum r", "pearson_mean", ".6f"),
        ("R^2", "r2", ".6f"),
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
# Forward: all-model workflow
# ==================================================


def train_all_models(
    *,
    operator_specs: list[dict[str, object]] | None = None,
    num_configurations: int = 500,
    batch_size: int = 16,
    dataset_file: str = DATASET_FILE,
    regenerate_dataset: bool = False,
    seed: int = SEED,
    lbfgs_epochs: int = 0,
    epochs_override: int | None = None,
) -> dict[str, object]:
    """Train and evaluate the given operators sequentially (default: all of them).

    ``operator_specs`` is a list of registry spec dicts (``OPERATORS[key]``
    values); pass e.g. ``[OPERATORS["2"], OPERATORS["4"], OPERATORS["5"]]``
    to train only DNO/GNO/STO instead of the full lineup. Defaults to every
    registered operator when omitted, matching the previous "always train
    all" behavior.

    ``epochs_override``, when given, applies to every selected operator
    instead of each one's own ``DEFAULT_MODEL_CONFIG``-adjacent default
    (``None`` preserves the original per-operator-default behavior).

    This workflow is deliberately non-interactive for Matplotlib: every plot is
    saved to ``plots/<operator>/`` and immediately closed. No ``plt.show()`` is
    called, so training proceeds continuously without waiting for plot windows.
    """
    specs = operator_specs if operator_specs is not None else list(OPERATORS.values())
    if not specs:
        raise ValueError("operator_specs must not be empty.")

    comparison_rows: list[dict[str, object]] = []
    all_model_plot_data: dict[str, dict[str, np.ndarray]] = {}
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("TRAIN + EVALUATE SELECTED ERP NEURAL OPERATORS")
    print(f"Operators      : {', '.join(spec['short'] for spec in specs)}")
    print(f"Dataset        : {dataset_file}")
    print(f"Configurations : {num_configurations}")
    print(f"Batch size     : {batch_size}")
    print(f"Seed           : {seed}")
    print(f"Regenerate     : {regenerate_dataset}")
    print(f"L-BFGS steps   : {lbfgs_epochs} (after AdamW; 0 = disabled)")
    if epochs_override is not None:
        print(f"Epochs         : {epochs_override} (override, applied to every operator)")
    else:
        print("Epochs         : each operator's own default (see per-operator lines below)")
    print(f"Plots folder   : {PLOTS_DIR.resolve()}")
    print(f"Plot pipeline  : {PLOT_PIPELINE_VERSION}")
    print("Interactive plots: disabled for uninterrupted batch training")
    print("=" * 76)

    for model_index, spec in enumerate(specs):
        epochs = int(epochs_override) if epochs_override is not None else int(spec["epochs"])
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
            lbfgs_epochs=lbfgs_epochs,
            dataset_file=dataset_file,
            regenerate_dataset=regenerate_this_model,
            seed=seed,
            plot=False,
            save_plots=False,           # cli.py saves explicitly below
            plots_dir=PLOTS_DIR,
            num_evaluation_plots=5,
            evaluate_after_training=True,
        )

        history = result["history"]
        metrics = result["metrics"]

        # Explicit cli -> plotting.py call.  This is intentionally outside the
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
                # evaluate_operator() uses these metric names:
                "pearson_global": metrics["pearson_correlation"],
                "pearson_mean": metrics["mean_spectrum_pearson_correlation"],
                "r2": metrics["r2_score"],
                "peak_frequency_mae_hz": metrics["peak_frequency_mae_hz"],
                "peak_amplitude_mae_db": metrics["peak_amplitude_mae_db"],
            }
        )

        # Keep only compact arrays needed for cross-architecture plots.
        # This avoids retaining the trained model or complete experiment object.
        pred = np.asarray(metrics["predictions"], dtype=np.float64)
        true = np.asarray(metrics["targets"], dtype=np.float64)
        error = pred - true

        spectrum_rmse = np.sqrt(np.mean(error**2, axis=1))
        spectrum_mae = np.mean(np.abs(error), axis=1)

        true_centered = true - np.mean(true, axis=1, keepdims=True)
        pred_centered = pred - np.mean(pred, axis=1, keepdims=True)
        corr_num = np.sum(true_centered * pred_centered, axis=1)
        corr_den = np.sqrt(
            np.sum(true_centered**2, axis=1)
            * np.sum(pred_centered**2, axis=1)
        )
        spectrum_pearson = np.divide(
            corr_num,
            corr_den,
            out=np.full(corr_num.shape, np.nan, dtype=np.float64),
            where=corr_den > 0.0,
        )

        freq = np.asarray(result["dataset"].frequency_values, dtype=np.float64)
        true_peak_idx = np.argmax(true, axis=1)
        pred_peak_idx = np.argmax(pred, axis=1)
        peak_frequency_abs_error = np.abs(
            freq[pred_peak_idx] - freq[true_peak_idx]
        )
        peak_amplitude_abs_error = np.abs(
            pred[np.arange(pred.shape[0]), true_peak_idx]
            - true[np.arange(true.shape[0]), true_peak_idx]
        )

        all_model_plot_data[spec["short"]] = {
            "train_loss": np.asarray(history["train"], dtype=np.float64),
            "val_loss": np.asarray(history["val"], dtype=np.float64),
            "spectrum_rmse": spectrum_rmse,
            "spectrum_mae": spectrum_mae,
            "spectrum_pearson": spectrum_pearson,
            "peak_frequency_abs_error": peak_frequency_abs_error,
            "peak_amplitude_abs_error": peak_amplitude_abs_error,
        }

        # Release the complete experiment result before the next architecture.
        del history, metrics, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_comparison_table(comparison_rows)

    comparison_plot_dir = save_all_model_comparison_plots(
        all_model_plot_data,
        comparison_rows,
        plots_dir=PLOTS_DIR,
        show=False,
    )

    return {
        "action": "train_all",
        "comparison": comparison_rows,
        "plots_dir": str(PLOTS_DIR),
        "comparison_plots_dir": str(comparison_plot_dir),
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

    operator_specs = _prompt_operator_selection(
        "Which operators to train (e.g. 2,4,5)"
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
    epochs_override = _prompt_optional_int(
        "Epochs to use for every selected operator (overrides each one's own default)",
        minimum=1,
    )
    regenerate_dataset = _prompt_yes_no(
        "Regenerate and overwrite the ERP dataset before training",
        default=False,
    )
    lbfgs_epochs = _prompt_lbfgs_epochs()

    print(f"All figures will be saved under: {PLOTS_DIR}")
    print("No figures will be displayed during all-model training.")

    return train_all_models(
        operator_specs=operator_specs,
        num_configurations=num_configurations,
        batch_size=batch_size,
        dataset_file=DATASET_FILE,
        regenerate_dataset=regenerate_dataset,
        seed=SEED,
        lbfgs_epochs=lbfgs_epochs,
        epochs_override=epochs_override,
    )


# ==================================================
# Forward: single-operator workflow
# ==================================================


def main_forward():
    """Interactive entry point for the forward (configuration -> ERP) workflow."""
    _print_operator_menu()
    all_models_key = str(len(OPERATORS) + 1)
    operator_choices = {**OPERATORS, all_models_key: None}
    operator_key = _prompt_choice("Select operator/workflow: ", operator_choices)

    if operator_key == all_models_key:
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
        lbfgs_epochs = _prompt_lbfgs_epochs()
        show_plot = _prompt_yes_no(
            "Also display the saved training and evaluation plots", default=False
        )

        # Train one operator and evaluate it immediately afterwards so a
        # standalone model run saves the same diagnostic plots as the
        # all-model workflow. Plot creation is kept in cli.py explicitly.
        result = runner(
            action="train",
            num_configurations=num_configurations,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            lbfgs_epochs=lbfgs_epochs,
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
            dataset_file=DATASET_FILE_OVERRIDES.get(operator_key, DATASET_FILE),
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
        dataset_file=DATASET_FILE_OVERRIDES.get(operator_key, DATASET_FILE),
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


# ==================================================
# Inverse: all-models workflow
# ==================================================


def main_all_inverse_models():
    """Train every inverse model, then optionally run the solver-scored evaluations."""
    print("\nAll-inverse-model training parameters")
    print("Per-model defaults come from inverse_operators.registry.INVERSE_MODELS.")
    for spec in INVERSE_MODELS.values():
        print(f"  {spec['short']:>10s}: epochs={spec['epochs']}, lr={spec['lr']:g}")

    num_configurations = _prompt_int(
        "Number of configurations to use", default=10000, minimum=3
    )
    batch_size = _prompt_int(
        "Batch size (target spectra per batch)", default=64, minimum=1
    )
    epochs = _prompt_int(
        "Epochs (applied to every inverse model)", default=150, minimum=1
    )

    from inverse_operators.train_all import main as train_all_inverse

    print(f"\nTraining all {len(INVERSE_MODELS)} inverse models, then building the "
          "solver-scored validation figure + per-sample report ...")
    train_all_inverse(
        num_configurations=num_configurations,
        epochs=epochs,
        batch_size=batch_size,
        dataset_file=INVERSE_DATASET_FILE,
    )

    if _prompt_yes_no(
        "\nAlso run the aggregate prediction-quality evaluation "
        "(MAE/RMSE/Pearson r/R^2 + prediction-vs-ground-truth, scored with "
        "the actual solver)?",
        default=True,
    ):
        num_test_examples = _prompt_int(
            "Held-out target spectra to evaluate", default=150, minimum=1
        )
        num_samples = _prompt_int(
            "Samples per target (solver calls scale as examples x samples)",
            default=8, minimum=1,
        )
        inverse_evaluate.main(
            num_test_examples=num_test_examples,
            num_samples=num_samples,
            num_configurations=num_configurations,
            dataset_file=INVERSE_DATASET_FILE
            if isinstance(INVERSE_DATASET_FILE, (list, tuple))
            else (INVERSE_DATASET_FILE,),
        )

    if _prompt_yes_no(
        "\nAlso run design-parameter recovery evaluation "
        "(predicted vs. true m, k, x, y)?",
        default=True,
    ):
        num_test_examples = _prompt_int(
            "Held-out target spectra to evaluate", default=150, minimum=1
        )
        num_samples = _prompt_int(
            "Samples per target", default=8, minimum=1
        )
        inverse_evaluate_design.main(
            num_test_examples=num_test_examples,
            num_samples=num_samples,
            num_configurations=num_configurations,
            dataset_file=INVERSE_DATASET_FILE
            if isinstance(INVERSE_DATASET_FILE, (list, tuple))
            else (INVERSE_DATASET_FILE,),
        )


# ==================================================
# Inverse: single-model workflow
# ==================================================


def main_inverse():
    """Interactive entry point for the inverse (ERP -> configuration) workflow."""
    _print_inverse_menu()
    all_models_key = str(len(INVERSE_MODELS) + 1)
    choices = {**INVERSE_MODELS, all_models_key: None}
    key = _prompt_choice("Select inverse model/workflow: ", choices)

    if key == all_models_key:
        return main_all_inverse_models()

    spec = INVERSE_MODELS[key]
    print("\n" + "=" * 68)
    print(f"Model    : {spec['name']} ({spec['short']})")
    print(f"Dataset  : {INVERSE_DATASET_FILE}")
    print(f"Seed     : {SEED}")
    print("Only 'train' is available for a single inverse model -- the")
    print("solver-scored evaluations compare all 4 models against each")
    print("other and live under 'Train ALL inverse models' instead.")
    print("=" * 68)

    num_configurations = _prompt_int(
        "Number of configurations to use", default=10000, minimum=3
    )
    batch_size = _prompt_int(
        "Batch size (target spectra per batch)", default=64, minimum=1
    )
    epochs = _prompt_int(
        "Number of epochs", default=int(spec["epochs"]), minimum=1
    )

    model, history, dataset = train_one_inverse_model(
        key,
        num_configurations=num_configurations,
        epochs=epochs,
        batch_size=batch_size,
        dataset_file=INVERSE_DATASET_FILE,
        seed=SEED,
    )
    print(
        f"\n{spec['short']} trained: final val loss={history['val'][-1]:.4f}, "
        f"best val loss={min(history['val']):.4f}"
    )
    print(f"Checkpoint saved to models/inverse_{spec['short'].lower()}.pth")
    return {"model": model, "history": history, "dataset": dataset}


# ==================================================
# Top level
# ==================================================


def main():
    """Interactive entry point: choose forward or inverse, then dispatch."""
    print("\nWhat would you like to run?")
    print("1. Forward  (resonator configuration -> ERP spectrum)")
    print("2. Inverse  (ERP spectrum -> resonator configuration)")
    mode = _prompt_choice("Select workflow: ", {"1": None, "2": None})
    return main_forward() if mode == "1" else main_inverse()
