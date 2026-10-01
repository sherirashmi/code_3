"""Error diagnostics of every trained forward operator -- no retraining.

Re-scores each saved ``models/GENERAL/<dataset>/*.pth`` checkpoint on its
test split and reports where the error sits (see
``neural_operator_utils.spectrum_error_diagnostics``):

* RMSE in the lowest / interior / highest 5% of the frequency axis
  (edge effects, e.g. the FNO's periodic FFT wrap),
* RMSE at the true resonance peaks vs off the peaks,
* RMSE on the 10% of configurations with the closest pair of tuning
  frequencies and the 10% with a resonator nearest a plate edge.

Writes ``plots/GENERAL/<dataset>/ALL_MODELS/error_breakdown.txt`` (+ .json)
and ``error_breakdown_bars.png``.

Usage::

    python -m erp_forward_operators.diagnose            # 100k models
    python -m erp_forward_operators.diagnose 10k
"""

from __future__ import annotations

import json
import sys

import numpy as np

from erp_forward_operators.neural_operator_utils import (
    build_operator_model,
    device,
    evaluate_operator,
    load_operator_checkpoint,
    prepare_operator_data,
)
from erp_forward_operators.operator_registry import OPERATORS
from utils.erp_dataset import DATASETS, select_dataset_modal_resolution
from utils.paths import ALL_MODELS, FORWARD_ROOT, GENERAL, forward_plot_root
from utils.plotting import plot_error_breakdown

_SPEC_BY_SHORT = {str(spec["short"]): spec for spec in OPERATORS.values()}

_COLUMNS = (
    ("Model", "model", None),
    ("RMSE", "rmse", ".3f"),
    ("Low 5% f", "rmse_low_band_db", ".3f"),
    ("Interior", "rmse_interior_db", ".3f"),
    ("High 5% f", "rmse_high_band_db", ".3f"),
    ("At peaks", "rmse_at_peaks_db", ".3f"),
    ("Off peaks", "rmse_off_peaks_db", ".3f"),
    ("Close f_t", "rmse_close_ft_db", ".3f"),
    ("Rest", "rmse_not_close_ft_db", ".3f"),
    ("Near edge", "rmse_near_edge_db", ".3f"),
    ("Rest ", "rmse_not_near_edge_db", ".3f"),
)


def _display_name(stem: str) -> str:
    """``dno_sorted_phys`` -> ``DNO_sorted_phys`` (the name the CLI uses)."""
    head, _, rest = stem.partition("_")
    return head.upper() + (f"_{rest}" if rest else "")


def _format_table(rows: list[dict[str, object]]) -> str:
    cells = [
        [str(r[k]) if fmt is None else format(float(r[k]), fmt) for _, k, fmt in _COLUMNS]
        for r in rows
    ]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, (h, _, _) in enumerate(_COLUMNS)]
    lines = [
        " | ".join(h.ljust(widths[i]) for i, (h, _, _) in enumerate(_COLUMNS)),
        "-+-".join("-" * w for w in widths),
        *(" | ".join(v.ljust(widths[i]) for i, v in enumerate(c)) for c in cells),
    ]
    return "\n".join(lines)


def main(dataset_tag: str = "100k", batch_size: int = 128) -> list[dict[str, object]]:
    if dataset_tag not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_tag}'; choose from {sorted(DATASETS)}.")
    select_dataset_modal_resolution(dataset_tag)
    model_dir = FORWARD_ROOT / "models" / GENERAL / dataset_tag
    checkpoints = sorted(model_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No forward checkpoints in {model_dir}.")

    rows: list[dict[str, object]] = []
    loaded_key = None
    dataset = loaders = None
    for path in checkpoints:
        checkpoint = load_operator_checkpoint(str(path))
        name = _display_name(path.stem)
        spec = _SPEC_BY_SHORT.get(name.partition("_")[0])
        if spec is None:
            print(f"Skipping {path.name}: no registered architecture for it.")
            continue
        state = checkpoint["preprocessing_state"]
        selected = np.asarray(state["selected_source_ids"], dtype=np.int64)
        key = (str(state.get("dataset_file")), selected.size, int(selected[:16].sum()))
        if key != loaded_key:  # models trained on the same selection share one load
            dataset, loaders = prepare_operator_data(
                num_configurations=int(selected.size),
                batch_size=batch_size,
                dataset_file=state.get("dataset_file") or DATASETS[dataset_tag]["files"],
                preprocessing_state=state,
                verbose=False,
            )
            loaded_key = key
        try:
            model = build_operator_model(spec["build_model"], dataset.num_res, checkpoint["model_config"])
            model.load_state_dict(checkpoint["model_state_dict"])
        except (TypeError, RuntimeError) as exc:
            print(f"Skipping {name}: checkpoint does not match the current code ({type(exc).__name__}).")
            continue
        print(f"\n### {name}")
        metrics = evaluate_operator(
            model.to(device), loaders, dataset.norm_params, dataset.frequency_values,
            plot=False, save_plots=False, operator_name=name,
        )
        rows.append({"model": name, "rmse": metrics["rmse"], **metrics["diagnostics"]})

    if not rows:
        raise RuntimeError("No checkpoint could be evaluated.")
    out_dir = forward_plot_root(dataset_tag, GENERAL) / ALL_MODELS
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _format_table(rows)
    print("\n" + table)
    (out_dir / "error_breakdown.txt").write_text(
        f"Test-error breakdown, dataset {dataset_tag} (RMSE in dB)\n\n{table}\n"
    )
    (out_dir / "error_breakdown.json").write_text(json.dumps(rows, indent=2))
    plot_error_breakdown(rows, save_path=out_dir / "error_breakdown_bars.png", title_suffix=f" ({dataset_tag})")
    print(f"\nSaved error breakdown to {out_dir}")
    return rows


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["100k"]))
