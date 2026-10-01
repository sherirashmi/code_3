"""Interactive CLI for this project's neural-operator problems.

``main.py`` at the repo root is a thin entry point; every prompt, menu, and
orchestration function lives here. Three top-level problems:

- **ERP** (resonator configuration <-> ERP spectrum):
    - **Forward** (configuration -> ERP spectrum):
        1. general train / evaluate / predict for any operator in
           ``erp_forward_operators.operator_registry.OPERATORS`` (or all), or
        2. the frequency-holdout generalisation experiment.
    - **Inverse** (ERP spectrum -> configuration): train / evaluate / predict
      for any model in ``erp_inverse_operators.registry.INVERSE_MODELS``, or
      train all of them (+ solver-scored cohort evaluation).
- **Displacement field**: ``displacement_forward_operators.train_all.run_one``.
- **Invertible** (iFNO / iDCO / iGNO, joint forward + inverse): train or
  evaluate, see ``erp_invertible_operators/train.py``.

Every ERP workflow first asks WHICH DATASET to use (``utils.erp_dataset.
DATASETS``). The choice fixes (a) the plate-mode basis the solver uses for
every reference spectrum (15x10 = 150 modes, or 6x3 = 18 modes for the
18-mode dataset -- applied automatically when the dataset is loaded), and
(b) the dataset sub-folder all checkpoints and plots go into, see
``utils/paths.py``:

    erp_forward_operators/plots/GENERAL/<dataset>/<MODEL>/
        loss_curve.png, erp_spectrum_test_config_01..05.png,
        prediction_vs_ground_truth.png, prediction_spectrum.png
    erp_forward_operators/plots/GENERAL/<dataset>/ALL_MODELS/
        all_training_loss.png, all_validation_loss.png, all_models_loss.png, ...
"""

from __future__ import annotations

import gc
import traceback
from pathlib import Path

import numpy as np
import torch

from utils.erp_dataset import (
    DATASETS,
    DEFAULT_DATASET_TAG,
    dataset_files,
    dataset_tag_for,
    select_dataset_modal_resolution,
)
from utils.paths import (
    ALL_MODELS,
    FREQ_HOLDOUT,
    GENERAL,
    INVERSE_ROOT,
    INVERTIBLE_ROOT,
    forward_model_path,
    forward_plot_dir,
    forward_plot_root,
    inverse_model_path,
    inverse_plot_dir,
)
from erp_forward_operators.operator_registry import OPERATORS, SORTED_SUFFIX
from erp_forward_operators.frequency_holdout import run_frequency_holdout
from utils.physics import Lx, Ly, fmin, fmax, m_min, m_max, num_res as default_num_res
from utils.plotting import (
    save_all_model_comparison_plots,
    save_operator_experiment_plots,
)

from erp_inverse_operators.registry import INVERSE_MODELS
from erp_inverse_operators.train_all import train_one_inverse_model
from erp_inverse_operators import evaluate as inverse_evaluate
from erp_inverse_operators import evaluate_design as inverse_evaluate_design

from displacement_forward_operators.operator_registry import OPERATORS as DISPLACEMENT_OPERATORS

from erp_invertible_operators.registry import INVERTIBLE_OPERATORS


SEED = 727

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


def _prompt_int(prompt: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return int(default)
        try:
            value = int(raw)
            if value >= minimum and (maximum is None or value <= maximum):
                return value
        except ValueError:
            pass
        bound = f" and <= {maximum}" if maximum is not None else ""
        print(f"Please enter an integer >= {minimum}{bound}.")


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


def _prompt_float(prompt: str, default: float, minimum: float = 0.0, maximum: float | None = None) -> float:
    """Value must be > minimum (and <= maximum if given)."""
    while True:
        raw = input(f"{prompt} [{default:g}]: ").strip()
        if not raw:
            return float(default)
        try:
            value = float(raw)
            if value > minimum and (maximum is None or value <= maximum):
                return value
        except ValueError:
            pass
        bound = f" and <= {maximum:g}" if maximum is not None else ""
        print(f"Please enter a number > {minimum:g}{bound}.")


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


def _dataset_available(tag: str) -> bool:
    return all(Path(f).exists() for f in DATASETS[tag]["files"])


def _prompt_dataset(default_tag: str = DEFAULT_DATASET_TAG) -> tuple[str, str | list[str]]:
    """Ask which raw dataset to use -> (tag, file or shard list).

    The tag decides the solver's plate-mode basis (applied automatically
    when the data is loaded) and the dataset sub-folder for checkpoints and
    plots.
    """
    tags = list(DATASETS)
    print("\nWhich dataset?")
    for i, tag in enumerate(tags, start=1):
        missing = "" if _dataset_available(tag) else "   [files missing]"
        print(f"{i}. {tag:<13s} {DATASETS[tag]['label']}{missing}")
    default_index = tags.index(default_tag) + 1 if default_tag in tags else 1
    while True:
        index = _prompt_int("Select dataset", default=default_index, minimum=1, maximum=len(tags))
        tag = tags[index - 1]
        if _dataset_available(tag) or tag == "10k":  # the 10k file can be (re)generated
            # Fix Nx/Ny now (15 x 10 for 150 modes, 6 x 3 for 18 modes) so every
            # calculation in this run -- training, solver references, and any
            # model trained on this dataset -- uses the dataset's modal basis.
            select_dataset_modal_resolution(tag)
            return tag, dataset_files(tag)
        print(f"Dataset '{tag}' is missing files: {DATASETS[tag]['files']}")


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


def _parse_selection(raw: str, registry: dict) -> list[str] | None:
    """Parse "2,4,5" / "2 4 5" into registry keys (order kept, repeats dropped).

    Blank -> None (= all). Any invalid token -> None too, so the caller can
    distinguish via ``raw.strip()`` and re-prompt; never raises.
    """
    raw = raw.strip()
    if not raw:
        return None
    tokens = [t for t in raw.replace(",", " ").split() if t]
    if any(t not in registry for t in tokens):
        return None
    seen: set[str] = set()
    return [t for t in tokens if not (t in seen or seen.add(t))]


def _parse_operator_selection(raw: str) -> list[dict[str, object]] | None:
    keys = _parse_selection(raw, OPERATORS)
    return None if keys is None else [OPERATORS[k] for k in keys]


def _prompt_selection(prompt: str, registry: dict) -> list[str]:
    menu = ", ".join(f"{key}={spec['short']}" for key, spec in registry.items())
    while True:
        raw = input(f"{prompt} [blank = all -- {menu}]: ")
        selected = _parse_selection(raw, registry)
        if raw.strip() and selected is None:
            print(f"Unrecognized number(s) in '{raw.strip()}'. Valid keys: {', '.join(registry)}")
            continue
        return selected if selected is not None else list(registry)


def _prompt_operator_selection(prompt: str) -> list[dict[str, object]]:
    return [OPERATORS[k] for k in _prompt_selection(prompt, OPERATORS)]


def _prompt_num_configurations(tag: str, prompt: str = "Number of configurations to use") -> int:
    total = int(DATASETS[tag]["num_configurations"])
    return _prompt_int(f"{prompt} (dataset has {total:,})", default=total, minimum=3, maximum=total)


def _prompt_configuration(num_res: int) -> np.ndarray:
    """Read one raw configuration in [m, k, f_t, x, y] order.

    ``m`` and ``f_t`` are entered; stiffness ``k = m*(2*pi*f_t)**2`` is
    derived, matching how the dataset itself is generated.
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

            if not np.isfinite([m, f_t, x, y]).all() or m <= 0.0:
                print("Mass must be a positive, finite number.")
                continue
            if not (fmin <= f_t <= fmax):
                print(f"f_t must be between {fmin:g} and {fmax:g} Hz.")
                continue
            if not (0.0 <= x <= Lx and 0.0 <= y <= Ly):
                print(f"Position must satisfy 0 <= x <= {Lx:g}, 0 <= y <= {Ly:g}.")
                continue
            if not (m_min <= m <= m_max):
                print(f"Note: m={m:g} kg is outside the training range [{m_min:g}, {m_max:g}] kg "
                      "-- the prediction will be an extrapolation.")

            k_val = m * (2.0 * np.pi * f_t) ** 2
            configuration[i] = [m, k_val, f_t, x, y]
            print(f"Derived stiffness: {k_val:.3f} N/m")
            break

    print("\nConfiguration used ([m, k, f_t, x, y] per resonator):")
    print(configuration)
    return configuration


def _variant(spec, use_sorted_branch: bool) -> tuple[str, dict[str, object]]:
    """(model name, model_config overrides) for the chosen encoder variant.

    With the sorted branch the model is saved/plotted as ``<SHORT>_sorted``
    (e.g. ``DCO_sorted``), so it never overwrites the set-encoder model.
    Architectures without a set encoder (GNO, STO, NN) always use their
    standard encoder.
    """
    if use_sorted_branch and spec.get("supports_sorted_branch"):
        return f"{spec['short']}{SORTED_SUFFIX}", {"use_sorted_branch": True}
    return spec["short"], {}


def _prompt_encoder_variant(names_supported: list[str]) -> bool:
    """Ask for the resonator-encoder variant; False = set encoder only."""
    if not names_supported:
        return False
    print("\nResonator encoder")
    print("1. Set encoder (mean + max pooling)  [standard]")
    print("2. Set encoder + f_t-sorted branch   (DCO_sorted design; saved as <MODEL>_sorted)")
    print(f"   (available for: {', '.join(names_supported)})")
    return _prompt_choice("Select encoder: ", {"1": None, "2": None}) == "2"


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================================================
# Menus / reporting
# ==================================================


def _print_operator_menu() -> None:
    print("\nAvailable forward neural operators")
    print("=" * 52)
    for key, spec in OPERATORS.items():
        print(f"{key}. {spec['name']} ({spec['short']})")
    print(f"{len(OPERATORS) + 1}. Train and evaluate ALL (or a selection of) forward operators")
    print("=" * 52)


def _print_inverse_menu() -> None:
    print("\nAvailable inverse (ERP -> design) models")
    print("=" * 52)
    for key, spec in INVERSE_MODELS.items():
        print(f"{key}. {spec['name']} ({spec['short']})")
    print(f"{len(INVERSE_MODELS) + 1}. Train ALL (or a selection of) inverse models "
          "(+ optional solver-scored evaluation)")
    print("=" * 52)


def _print_action_menu() -> None:
    print("\nOperation")
    print("1. Train    (then evaluate on the test split and save all plots)")
    print("2. Evaluate (saved checkpoint, test split)")
    print("3. Predict  (one configuration, compared with the solver)")


_TABLE_COLUMNS = [
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


def _format_comparison_table(rows: list[dict[str, object]]) -> str:
    formatted_rows = [
        [str(row[key]) if fmt is None else format(float(row[key]), fmt) for _, key, fmt in _TABLE_COLUMNS]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(r[idx]) for r in formatted_rows))
        for idx, (header, _, _) in enumerate(_TABLE_COLUMNS)
    ]
    lines = [
        " | ".join(header.ljust(widths[i]) for i, (header, _, _) in enumerate(_TABLE_COLUMNS)),
        "-+-".join("-" * w for w in widths),
    ]
    lines += [" | ".join(v.ljust(widths[i]) for i, v in enumerate(r)) for r in formatted_rows]
    return "\n".join(lines)


def _print_comparison_table(rows: list[dict[str, object]]) -> str:
    """Print (and return) a dependency-free final comparison table."""
    table = _format_comparison_table(rows)
    width = len(table.splitlines()[0])
    print("\n" + "=" * width)
    print("FINAL ALL-MODEL COMPARISON")
    print("=" * width)
    print(table)
    print("=" * width)
    return table


# ==================================================
# Forward: shared helpers
# ==================================================


def _comparison_entry(name, history, metrics, frequency_values):
    """Summary row + compact per-spectrum arrays for the ALL_MODELS plots."""
    row = {
        "model": name,
        "final_train_loss": history["train"][-1],
        "final_val_loss": history["val"][-1],
        "best_val_loss": min(history["val"]),
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "pearson_global": metrics["pearson_correlation"],
        "pearson_mean": metrics["mean_spectrum_pearson_correlation"],
        "r2": metrics["r2_score"],
        "peak_frequency_mae_hz": metrics["peak_frequency_mae_hz"],
        "peak_amplitude_mae_db": metrics["peak_amplitude_mae_db"],
    }

    pred = np.asarray(metrics["predictions"], dtype=np.float64)
    true = np.asarray(metrics["targets"], dtype=np.float64)
    error = pred - true
    true_centered = true - true.mean(axis=1, keepdims=True)
    pred_centered = pred - pred.mean(axis=1, keepdims=True)
    corr_num = np.sum(true_centered * pred_centered, axis=1)
    corr_den = np.sqrt(np.sum(true_centered**2, axis=1) * np.sum(pred_centered**2, axis=1))
    freq = np.asarray(frequency_values, dtype=np.float64)
    true_peak_idx = np.argmax(true, axis=1)
    pred_peak_idx = np.argmax(pred, axis=1)
    rows = np.arange(pred.shape[0])
    data = {
        "train_loss": np.asarray(history["train"], dtype=np.float64),
        "val_loss": np.asarray(history["val"], dtype=np.float64),
        "spectrum_rmse": np.sqrt(np.mean(error**2, axis=1)),
        "spectrum_mae": np.mean(np.abs(error), axis=1),
        "spectrum_pearson": np.divide(
            corr_num, corr_den, out=np.full(corr_num.shape, np.nan), where=corr_den > 0.0
        ),
        "peak_frequency_abs_error": np.abs(freq[pred_peak_idx] - freq[true_peak_idx]),
        "peak_amplitude_abs_error": np.abs(pred[rows, true_peak_idx] - true[rows, true_peak_idx]),
    }
    return row, data


def _require_checkpoint(name: str, tag: str) -> Path | None:
    path = forward_model_path(name, tag)
    if path.exists():
        return path
    general_dir = path.parent.parent
    available = sorted(p.parent.name for p in general_dir.glob(f"*/{path.name}"))
    print(f"\nNo {name} checkpoint for dataset '{tag}' ({path}).")
    print(f"Datasets {name} has been trained on: {', '.join(available) or 'none'} -- train it first.")
    return None


# ==================================================
# Forward: all-model workflow
# ==================================================


def train_all_models(
    *,
    operator_specs: list[dict[str, object]] | None = None,
    num_configurations: int = 500,
    batch_size: int = 16,
    dataset_file: str | list[str] = DATASETS[DEFAULT_DATASET_TAG]["files"][0],
    regenerate_dataset: bool = False,
    seed: int = SEED,
    lbfgs_epochs: int = 0,
    epochs_override: int | None = None,
    use_sorted_branch: bool = False,
) -> dict[str, object]:
    """Train and evaluate the given operators sequentially (default: all).

    ``use_sorted_branch=True`` trains the set-encoder architectures with the
    extra f_t-sorted resonator branch, saved as ``<MODEL>_sorted``.

    For every operator, saves into ``plots/GENERAL/<dataset>/<MODEL>/`` its
    loss curve, 5 test-spectrum comparisons and the predicted-vs-true parity
    plot; then the cross-model figures (incl. all models' training and
    validation loss) into ``plots/GENERAL/<dataset>/ALL_MODELS/``.

    A failure in one operator (e.g. out of memory) is reported and the
    remaining operators still run, so an overnight run is not lost.
    """
    specs = operator_specs if operator_specs is not None else list(OPERATORS.values())
    if not specs:
        raise ValueError("operator_specs must not be empty.")
    tag = dataset_tag_for(dataset_file)
    plot_root = forward_plot_root(tag, GENERAL)

    comparison_rows: list[dict[str, object]] = []
    all_model_plot_data: dict[str, dict[str, np.ndarray]] = {}
    failures: dict[str, str] = {}

    print("\n" + "=" * 76)
    print("TRAIN + EVALUATE SELECTED ERP NEURAL OPERATORS")
    print(f"Operators      : {', '.join(_variant(spec, use_sorted_branch)[0] for spec in specs)}")
    print(f"Dataset        : {tag} ({dataset_file})")
    print(f"Configurations : {num_configurations}")
    print(f"Batch size     : {batch_size}")
    print(f"Seed           : {seed}")
    print(f"Regenerate     : {regenerate_dataset}")
    print(f"L-BFGS steps   : {lbfgs_epochs} (after AdamW; 0 = disabled)")
    if epochs_override is not None:
        print(f"Epochs         : {epochs_override} (override, applied to every operator)")
    else:
        print("Epochs         : each operator's own default (see per-operator lines below)")
    print(f"Plots folder   : {plot_root}/<MODEL>/ and {plot_root}/{ALL_MODELS}/")
    print("=" * 76)

    for model_index, spec in enumerate(specs):
        epochs = int(epochs_override) if epochs_override is not None else int(spec["epochs"])
        learning_rate = float(spec["lr"])
        regenerate_this_model = bool(regenerate_dataset and model_index == 0)
        name, overrides = _variant(spec, use_sorted_branch)

        print("\n" + "#" * 76)
        print(f"[{model_index + 1}/{len(specs)}] Training {spec['name']} ({name})")
        print(f"epochs={epochs} | lr={learning_rate:g}")
        print("#" * 76)

        try:
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
                save_plots=False,           # saved explicitly below
                num_evaluation_plots=5,
                evaluate_after_training=True,
                checkpoint_file=forward_model_path(name, tag),
                model_config_overrides=overrides,
            )
            history, metrics = result["history"], result["metrics"]
            save_operator_experiment_plots(
                name,
                history=history,
                metrics=metrics,
                frequency_values=result["dataset"].frequency_values,
                plot_dir=forward_plot_dir(name, tag),
                num_configurations=5,
                show=False,
            )
            row, data = _comparison_entry(name, history, metrics, result["dataset"].frequency_values)
            comparison_rows.append(row)
            all_model_plot_data[name] = data
            del history, metrics, result
        except Exception as exc:  # keep going with the other operators
            failures[name] = f"{type(exc).__name__}: {exc}"
            print(f"\n!!! {name} FAILED -- continuing with the remaining operators.")
            traceback.print_exc()
        _release_memory()

    comparison_plot_dir = None
    if comparison_rows:
        table = _print_comparison_table(comparison_rows)
        out_dir = plot_root / ALL_MODELS
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "comparison_table.txt").write_text(table + "\n")
        comparison_plot_dir = save_all_model_comparison_plots(
            all_model_plot_data,
            comparison_rows,
            out_dir=out_dir,
            title_suffix=f" ({tag})",
            show=False,
        )
    if failures:
        print("\nOperators that failed:")
        for name, message in failures.items():
            print(f"  {name}: {message}")

    return {
        "action": "train_all",
        "comparison": comparison_rows,
        "failures": failures,
        "plots_dir": str(plot_root),
        "comparison_plots_dir": str(comparison_plot_dir) if comparison_plot_dir else None,
    }


def main_all_models():
    """Interactive input collection for the non-blocking all-model workflow."""
    print("\nAll-model training parameters (per-model defaults):")
    for spec in OPERATORS.values():
        print(f"  {spec['short']:>5s}: epochs={int(spec['epochs'])}, lr={float(spec['lr']):g}")

    operator_specs = _prompt_operator_selection("Which operators to train (e.g. 2,4,5)")
    use_sorted_branch = _prompt_encoder_variant(
        [s["short"] for s in operator_specs if s.get("supports_sorted_branch")]
    )
    tag, dataset_file = _prompt_dataset()
    num_configurations = _prompt_num_configurations(tag, "Number of configurations to use for every model")
    batch_size = _prompt_int("Batch size (complete ERP spectra per batch)", default=64, minimum=1)
    epochs_override = _prompt_optional_int(
        "Epochs to use for every selected operator (overrides each one's own default)", minimum=1
    )
    if isinstance(dataset_file, list):
        regenerate_dataset = False
    else:
        regenerate_dataset = _prompt_yes_no(
            "Regenerate and overwrite the ERP dataset before training", default=False
        )
    lbfgs_epochs = _prompt_lbfgs_epochs()

    return train_all_models(
        operator_specs=operator_specs,
        num_configurations=num_configurations,
        batch_size=batch_size,
        dataset_file=dataset_file,
        regenerate_dataset=regenerate_dataset,
        seed=SEED,
        lbfgs_epochs=lbfgs_epochs,
        epochs_override=epochs_override,
        use_sorted_branch=use_sorted_branch,
    )


# ==================================================
# Forward: single-operator workflow
# ==================================================


def _print_forward_method_menu() -> None:
    print("\nWhich method?")
    print("1. General training  (train/evaluate/predict on the full frequency range)")
    print("2. Frequency-holdout generalization experiment  (mask a contiguous band of "
          "frequencies out of training entirely, then score seen vs. unseen frequencies)")


def main_forward():
    """Interactive entry point for the forward (configuration -> ERP) workflow."""
    _print_forward_method_menu()
    method = _prompt_choice("Select method: ", {"1": None, "2": None})
    if method == "2":
        return main_frequency_holdout()

    _print_operator_menu()
    all_models_key = str(len(OPERATORS) + 1)
    operator_key = _prompt_choice("Select operator/workflow: ", {**OPERATORS, all_models_key: None})
    if operator_key == all_models_key:
        return main_all_models()

    spec = OPERATORS[operator_key]
    _print_action_menu()
    action = ACTIONS[_prompt_choice("Select operation: ", ACTIONS)]
    runner = spec["runner"]
    use_sorted_branch = _prompt_encoder_variant([spec["short"]] if spec.get("supports_sorted_branch") else [])
    name, overrides = _variant(spec, use_sorted_branch)
    tag, dataset_file = _prompt_dataset("100k" if action != "train" else DEFAULT_DATASET_TAG)
    plot_dir = forward_plot_root(tag, GENERAL) / name  # created when plots are saved
    checkpoint = forward_model_path(name, tag)

    print("\n" + "=" * 68)
    print(f"Operator   : {spec['name']} ({name})")
    print(f"Action     : {action}")
    print(f"Dataset    : {tag}")
    print(f"Checkpoint : {checkpoint}")
    print(f"Plots      : {plot_dir}")
    print("=" * 68)

    if action == "train":
        num_configurations = _prompt_num_configurations(tag)
        batch_size = _prompt_int("Batch size (complete ERP spectra per batch)", default=16, minimum=1)
        epochs = _prompt_int("Number of epochs", default=int(spec["epochs"]), minimum=1)
        learning_rate = _prompt_float("Learning rate", default=float(spec["lr"]), minimum=0.0)
        regenerate_dataset = (
            False if isinstance(dataset_file, list)
            else _prompt_yes_no("Regenerate and overwrite the ERP dataset before training", default=False)
        )
        lbfgs_epochs = _prompt_lbfgs_epochs()
        show_plot = _prompt_yes_no("Also display the saved plots", default=False)

        result = runner(
            action="train",
            num_configurations=num_configurations,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            lbfgs_epochs=lbfgs_epochs,
            dataset_file=dataset_file,
            regenerate_dataset=regenerate_dataset,
            seed=SEED,
            plot=False,
            save_plots=False,
            num_evaluation_plots=5,
            evaluate_after_training=True,
            checkpoint_file=checkpoint,
            model_config_overrides=overrides,
        )
        save_operator_experiment_plots(
            name,
            history=result["history"],
            metrics=result["metrics"],
            frequency_values=result["dataset"].frequency_values,
            plot_dir=plot_dir,
            num_configurations=5,
            show=show_plot,
        )
        return result

    if _require_checkpoint(name, tag) is None:
        return None

    if action == "evaluate":
        batch_size = _prompt_int("Evaluation batch size", default=64, minimum=1)
        show_plot = _prompt_yes_no("Also display the saved plots", default=False)
        result = runner(
            action="evaluate",
            batch_size=batch_size,
            dataset_file=dataset_file,
            seed=SEED,
            plot=False,
            save_plots=False,
            num_evaluation_plots=5,
            checkpoint_file=checkpoint,
        )
        history = result.get("history")
        save_operator_experiment_plots(
            name,
            history=history if history and history.get("train") else None,
            metrics=result["metrics"],
            frequency_values=result["frequency_values"],
            plot_dir=plot_dir,
            num_configurations=5,
            show=show_plot,
        )
        return result

    print("\nPrediction configuration")
    print("1. Enter a new configuration manually")
    print("2. Use the first configuration from the saved test split")
    prediction_choice = _prompt_choice("Select prediction input: ", {"1": None, "2": None})
    configuration = _prompt_configuration(default_num_res) if prediction_choice == "1" else None
    show_plot = _prompt_yes_no("Also display the saved solver vs neural-operator ERP spectrum", default=False)

    # The dataset load inside the runner switches the solver to that
    # dataset's plate-mode basis, so the "ground truth" curve is computed
    # with the same physics the model was trained on.
    result = runner(
        action="predict",
        dataset_file=dataset_file,
        seed=SEED,
        configuration=configuration,
        plot=False,
        save_plots=False,
        checkpoint_file=checkpoint,
    )
    save_operator_experiment_plots(
        name,
        prediction=result["prediction"],
        plot_dir=plot_dir,
        show=show_plot,
    )
    return result


# ==================================================
# Forward: frequency-holdout generalization experiment
# ==================================================


def main_frequency_holdout():
    """Interactive entry point for the frequency-holdout experiment."""
    operator_specs = _prompt_operator_selection(
        "Which operators to run the frequency-holdout experiment on (e.g. 2,4,5)"
    )
    use_sorted_branch = _prompt_encoder_variant(
        [s["short"] for s in operator_specs if s.get("supports_sorted_branch")]
    )
    tag, dataset_file = _prompt_dataset()
    num_configurations = _prompt_num_configurations(tag)
    batch_size = _prompt_int("Batch size (complete ERP spectra per batch)", default=16, minimum=1)
    epochs = _prompt_int("Number of epochs (applied to every selected operator)", default=200, minimum=1)
    print(
        "\nThe held-out band is a contiguous slice of the frequency axis, given as "
        "fractions of the full range (default: the middle 20%, i.e. 0.40-0.60)."
    )
    while True:
        holdout_start_frac = _prompt_float("Holdout band start fraction", default=0.40, minimum=-1e-12, maximum=0.99)
        holdout_end_frac = _prompt_float("Holdout band end fraction", default=0.60, minimum=holdout_start_frac, maximum=1.0)
        if holdout_end_frac - holdout_start_frac >= 0.01:
            break
        print("The band must cover at least 1% of the frequency range.")
    num_plot = _prompt_int("Example test spectra to plot per operator", default=5, minimum=1)

    root = forward_plot_root(tag, FREQ_HOLDOUT)
    print("\n" + "=" * 68)
    print(f"Operators : {', '.join(spec['short'] for spec in operator_specs)}")
    print(f"Dataset   : {tag}")
    print(f"Holdout   : [{holdout_start_frac:.2f}, {holdout_end_frac:.2f}) of the frequency range")
    print(f"Plots     : {root}/<MODEL>/ and {root}/{ALL_MODELS}/")
    print(f"Models    : {forward_model_path('<model>', tag, FREQ_HOLDOUT)}")
    print("=" * 68)

    return run_frequency_holdout(
        operator_specs=operator_specs,
        num_configurations=num_configurations,
        epochs=epochs,
        batch_size=batch_size,
        dataset_file=dataset_file,
        holdout_start_frac=holdout_start_frac,
        holdout_end_frac=holdout_end_frac,
        num_plot=num_plot,
        seed=SEED,
        use_sorted_branch=use_sorted_branch,
    )


# ==================================================
# Inverse
# ==================================================


def _as_dataset_tuple(dataset_file) -> tuple:
    """evaluate.py/evaluate_design.py want a tuple/list; training wants str or list."""
    return tuple(dataset_file) if isinstance(dataset_file, (list, tuple)) else (dataset_file,)


def _run_inverse_evaluations(model_names, tag, dataset_file, num_configurations, *, ask: bool):
    dataset_tuple = _as_dataset_tuple(dataset_file)
    results = {}
    if not ask or _prompt_yes_no(
        "\nRun the solver-scored prediction-quality evaluation "
        "(MAE/RMSE/Pearson r/R^2 + predicted-vs-true ERP)?", default=True,
    ):
        num_test_examples = _prompt_int("Held-out target spectra to evaluate", default=150, minimum=1)
        num_samples = _prompt_int("Samples per target (solver calls scale as examples x samples)", default=8, minimum=1)
        results["stats"] = inverse_evaluate.main(
            num_test_examples=num_test_examples, num_samples=num_samples,
            num_configurations=num_configurations, dataset_file=dataset_tuple, model_names=model_names,
        )
        if not ask or _prompt_yes_no(
            "\nAlso run design-parameter recovery (predicted vs. true m, k, f_t, x, y)?", default=True,
        ):
            results["design_stats"] = inverse_evaluate_design.main(
                num_test_examples=num_test_examples, num_samples=num_samples,
                num_configurations=num_configurations, dataset_file=dataset_tuple, model_names=model_names,
            )
    return results


def main_all_inverse_models():
    """Train every (or a selection of) inverse model(s), then optionally evaluate."""
    print("\nAll-inverse-model training parameters (per-model defaults):")
    for spec in INVERSE_MODELS.values():
        print(f"  {spec['short']:>10s}: epochs={spec['epochs']}, lr={spec['lr']:g}")

    keys = _prompt_selection("Which inverse models to train (e.g. 1,3,7)", INVERSE_MODELS)
    tag, dataset_file = _prompt_dataset()
    num_configurations = _prompt_num_configurations(tag)
    batch_size = _prompt_int("Batch size (target spectra per batch)", default=64, minimum=1)
    epochs = _prompt_optional_int("Epochs for every selected model (overrides each one's own default)")

    from erp_inverse_operators.train_all import main as train_all_inverse

    print(f"\nTraining {len(keys)} inverse model(s); per-model loss curves -> "
          f"{INVERSE_ROOT / 'plots' / tag}/<MODEL>/, comparison figures -> {INVERSE_ROOT / 'plots' / tag / ALL_MODELS}")
    train_all_inverse(
        num_configurations=num_configurations,
        epochs=epochs,
        batch_size=batch_size,
        dataset_file=dataset_file,
        keys=keys,
        seed=SEED,
    )
    names = [INVERSE_MODELS[k]["short"] for k in keys]
    return _run_inverse_evaluations(names, tag, dataset_file, num_configurations, ask=True)


INVERSE_ACTIONS = {"1": "train", "2": "evaluate", "3": "predict"}


def _print_inverse_action_menu() -> None:
    print("\nOperation")
    print("1. Train")
    print("2. Evaluate (aggregate MAE/RMSE/Pearson r/R^2 + design-parameter recovery)")
    print("3. Predict (sample designs for one target spectrum)")


def main_inverse():
    """Interactive entry point for the inverse (ERP -> configuration) workflow."""
    _print_inverse_menu()
    all_models_key = str(len(INVERSE_MODELS) + 1)
    key = _prompt_choice("Select inverse model/workflow: ", {**INVERSE_MODELS, all_models_key: None})
    if key == all_models_key:
        return main_all_inverse_models()

    spec = INVERSE_MODELS[key]
    _print_inverse_action_menu()
    action = INVERSE_ACTIONS[_prompt_choice("Select operation: ", INVERSE_ACTIONS)]
    tag, dataset_file = _prompt_dataset("100k" if action != "train" else DEFAULT_DATASET_TAG)

    print("\n" + "=" * 68)
    print(f"Model      : {spec['name']} ({spec['short']})")
    print(f"Action     : {action}")
    print(f"Dataset    : {tag}")
    print(f"Checkpoint : {inverse_model_path(spec['short'], tag)}")
    print(f"Plots      : {INVERSE_ROOT / 'plots' / tag / spec['short']}")
    print("=" * 68)

    if action == "train":
        num_configurations = _prompt_num_configurations(tag)
        batch_size = _prompt_int("Batch size (target spectra per batch)", default=64, minimum=1)
        epochs = _prompt_int("Number of epochs", default=int(spec["epochs"]), minimum=1)
        model, history, dataset = train_one_inverse_model(
            key, num_configurations=num_configurations, epochs=epochs,
            batch_size=batch_size, dataset_file=dataset_file, seed=SEED,
        )
        print(f"\n{spec['short']} trained: final val loss={history['val'][-1]:.4f}, "
              f"best val loss={min(history['val']):.4f}")
        result = {"model": model, "history": history, "dataset": dataset}
        if _prompt_yes_no("Evaluate it now (solver-scored)?", default=True):
            result.update(_run_inverse_evaluations([spec["short"]], tag, dataset_file, num_configurations, ask=False))
        return result

    if not inverse_model_path(spec["short"], tag).exists():
        print(f"\nNo checkpoint {inverse_model_path(spec['short'], tag)} -- train {spec['short']} on '{tag}' first.")
        return None
    recorded = inverse_evaluate.checkpoint_num_configurations(spec["short"], tag)
    default_n = recorded or int(DATASETS[tag]["num_configurations"])
    num_configurations = _prompt_int(
        "Number of configurations backing the test split (must match training)",
        default=default_n, minimum=3, maximum=int(DATASETS[tag]["num_configurations"]),
    )

    if action == "evaluate":
        return _run_inverse_evaluations([spec["short"]], tag, dataset_file, num_configurations, ask=False)

    print("\nPrediction target")
    print("1. Enter a resonator configuration manually (its real solver-computed "
          "ERP spectrum becomes the target to invert)")
    print("2. Use the first spectrum from the saved test split")
    target_choice = _prompt_choice("Select target input: ", {"1": None, "2": None})
    configuration = _prompt_configuration(default_num_res) if target_choice == "1" else None
    num_samples = _prompt_int("Number of candidate designs to sample", default=6, minimum=1)

    from erp_inverse_operators.predict import predict_one, print_report

    result = predict_one(
        spec["short"],
        configuration=configuration,
        num_samples=num_samples,
        num_configurations=num_configurations,
        dataset_file=dataset_file,
        seed=SEED,
    )
    print_report(spec["short"], result)
    return result


# ==================================================
# Top level
# ==================================================


def main():
    """Interactive entry point: choose the problem, then dispatch."""
    print("\nWhich problem would you like to work on?")
    print("1. ERP           (resonator configuration <-> ERP spectrum)")
    print("2. Displacement  (configuration, frequency, position -> velocity field)")
    print("3. Invertible    (joint forward+inverse operator, ERP problem: iFNO/iDCO/iGNO)")
    problem = _prompt_choice("Select problem: ", {"1": None, "2": None, "3": None})
    if problem == "1":
        return main_erp()
    if problem == "2":
        return main_displacement()
    return main_invertible_operators()


def main_erp():
    """ERP problem: choose forward or inverse, then dispatch."""
    print("\nWhat would you like to run?")
    print("1. Forward  (resonator configuration -> ERP spectrum)")
    print("2. Inverse  (ERP spectrum -> resonator configuration)")
    mode = _prompt_choice("Select workflow: ", {"1": None, "2": None})
    return main_forward() if mode == "1" else main_inverse()


def _print_displacement_menu() -> None:
    print("\nAvailable displacement-field forward operators")
    print("=" * 52)
    for key, spec in DISPLACEMENT_OPERATORS.items():
        print(f"{key}. {spec['name']} ({spec['short']})")
    print(f"{len(DISPLACEMENT_OPERATORS) + 1}. Train and evaluate ALL displacement operators")
    print("=" * 52)


def main_displacement():
    """Interactive entry point for the displacement-field workflow (fixed,
    already-tuned pipeline per architecture, see
    ``displacement_forward_operators.train_all.run_one``)."""
    _print_displacement_menu()
    all_key = str(len(DISPLACEMENT_OPERATORS) + 1)
    key = _prompt_choice("Select architecture/workflow: ", {**DISPLACEMENT_OPERATORS, all_key: None})
    keys = list(DISPLACEMENT_OPERATORS.keys()) if key == all_key else [key]

    from displacement_forward_operators.train_all import run_one

    results = []
    for architecture_key in keys:
        try:
            name, result, train_time = run_one(architecture_key)
        except Exception:
            print(f"\n!!! displacement operator {architecture_key} FAILED -- continuing.")
            traceback.print_exc()
            continue
        print(
            f"\n{name}: RMSE={result['rmse']:.3f} dB  R^2={result['r2_score']:.3f}  "
            f"Pearson={result['pearson_correlation']:.3f}  ({train_time:.1f}s)"
        )
        results.append((name, result, train_time))
        _release_memory()
    return results


def _print_invertible_menu() -> None:
    print("\nAvailable invertible (joint forward+inverse) operators")
    print("=" * 56)
    for key, spec in INVERTIBLE_OPERATORS.items():
        print(f"{key}. {spec['name']} ({spec['short']})")
    print(f"{len(INVERTIBLE_OPERATORS) + 1}. ALL three")
    print("=" * 56)


def main_invertible_operators():
    """Invertible-operator family (Long et al., arXiv:2402.11722): train with
    the 3-stage schedule, or re-evaluate saved checkpoints."""
    import erp_invertible_operators.train as invertible

    _print_invertible_menu()
    all_key = str(len(INVERTIBLE_OPERATORS) + 1)
    key = _prompt_choice("Select architecture/workflow: ", {**INVERTIBLE_OPERATORS, all_key: None})
    keys = list(INVERTIBLE_OPERATORS.keys()) if key == all_key else [key]

    print("\nOperation")
    print("1. Train (3-stage schedule) + forward/inverse evaluation")
    print("2. Evaluate saved checkpoint(s)")
    evaluate_only = _prompt_choice("Select operation: ", {"1": None, "2": None}) == "2"
    use_sorted_branch = _prompt_encoder_variant(
        [INVERTIBLE_OPERATORS[k]["short"] for k in keys if INVERTIBLE_OPERATORS[k].get("supports_sorted_branch")]
    )
    tag, dataset_file = _prompt_dataset("100k")
    num_inverse_examples = _prompt_int("Held-out targets for the solver-scored inverse evaluation", default=100, minimum=1)
    num_inverse_samples = _prompt_int("Posterior samples per target", default=8, minimum=1)

    if evaluate_only:
        return invertible.main(
            keys=tuple(keys), evaluate_only=True, dataset_file=dataset_file, seed=SEED,
            use_sorted_branch=use_sorted_branch,
            num_inverse_examples=num_inverse_examples, num_inverse_samples=num_inverse_samples,
        )

    num_configurations = _prompt_num_configurations(tag)
    batch_size = _prompt_int("Batch size", default=invertible.BATCH_SIZE, minimum=1)
    print("\nThree-stage schedule (see erp_invertible_operators/train.py):")
    stage1 = _prompt_int("Stage 1 epochs (invertible blocks: J_FWD + J_INV + round-trip terms)",
                         default=invertible.STAGE1_EPOCHS, minimum=0)
    stage2 = _prompt_int("Stage 2 epochs (beta-VAE pretraining on designs)",
                         default=invertible.STAGE2_EPOCHS, minimum=0)
    stage3 = _prompt_int("Stage 3 epochs (joint fine-tuning)", default=invertible.STAGE3_EPOCHS, minimum=0)
    inverse_weight = _prompt_float("Stage 3 weight of the direct inverse term J_INV (0 = paper's eq 11)",
                                   default=invertible.STAGE3_INVERSE_WEIGHT, minimum=-1e-12)
    print(f"\nPlots -> {INVERTIBLE_ROOT / 'plots' / tag}/<MODEL>/ (and {ALL_MODELS}/ when training several)")
    return invertible.main(
        keys=tuple(keys),
        num_configurations=num_configurations,
        batch_size=batch_size,
        dataset_file=dataset_file,
        seed=SEED,
        stage1_epochs=stage1,
        stage2_epochs=stage2,
        stage3_epochs=stage3,
        stage3_inverse_weight=inverse_weight,
        use_sorted_branch=use_sorted_branch,
        num_inverse_examples=num_inverse_examples,
        num_inverse_samples=num_inverse_samples,
    )
