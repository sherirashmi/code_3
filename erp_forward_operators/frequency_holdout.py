"""Frequency-holdout generalization experiment for ERP forward operators.

Trains each selected operator with a contiguous BAND of frequencies masked
out of both the training and validation loaders entirely (never seen during
training at all, not just held out as extra test rows), then evaluates on
the FULL frequency range -- separately scoring the "seen" frequencies
(trained on) vs. the held-out "unseen" band (never trained on). The gap
between those two scores is a direct measure of whether each architecture
actually learned the underlying physics well enough to *interpolate* across
a frequency gap, or whether it only memorized pointwise input->output pairs
at the specific frequencies it was shown.

Generalized from the original one-off ``others/frequency_holdout_experiment.py``
(which hardcoded 6 operators + the old 10k dataset) into a reusable function
parameterized the same way this project's other CLI-driven workflows are,
so it can run against any operator subset and any dataset (including the
current 100k-configuration, [m,k,f_t,x,y]-schema one).

Output layout (see utils/paths.py), mirroring the general branch:

    erp_forward_operators/models/FREQ_HOLDOUT/<dataset>/<model>.pth
    erp_forward_operators/plots/FREQ_HOLDOUT/<dataset>/<MODEL>/
        loss_curve.png, erp_comparison_config_01..NN.png,
        prediction_vs_ground_truth.png
    erp_forward_operators/plots/FREQ_HOLDOUT/<dataset>/ALL_MODELS/
        seen_vs_unseen_rmse.png, all_models_loss.png, results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from erp_forward_operators.neural_operator_utils import (
    _configuration_features,
    _split_ids,
    device,
    parameter_count,
    prepare_operator_data,
    train_operator,
)
from erp_forward_operators.operator_registry import OPERATORS
from utils.erp_dataset import (
    dataset_tag_for,
    denormalize_configuration_array,
    denormalize_erp_array,
    normalize_configuration_array,
    normalize_erp_array,
    normalize_frequency_array,
)
from utils.paths import FREQ_HOLDOUT, forward_model_path, forward_plot_dir, forward_plot_root
from utils.plotting import (
    ERP_LABEL,
    plot_all_models_loss,
    plot_erp_comparison,
    plot_loss_curves,
    plot_prediction_scatter,
    save_figure,
)



class _MaskedFrequencyDataset(Dataset):
    """Same configuration/response rows as the full dataset, but restricted
    to only ``freq_indices`` of the frequency axis -- used to build loaders
    that never see the held-out band at all during training/validation.
    """

    def __init__(self, dataset, configuration_ids: np.ndarray, freq_indices: np.ndarray, norm) -> None:
        ids = np.asarray(configuration_ids, dtype=np.int64)
        configuration = normalize_configuration_array(_configuration_features(dataset)[ids], norm)
        frequency_full = normalize_frequency_array(dataset.frequency_values, norm)
        response_full = normalize_erp_array(
            np.asarray(dataset.responses, dtype=np.float32)[ids, :, 0], norm
        )
        self.configuration = torch.from_numpy(configuration.astype(np.float32))
        self.frequency = torch.from_numpy(frequency_full[freq_indices][:, None].astype(np.float32))
        self.response = torch.from_numpy(response_full[:, freq_indices][..., None].astype(np.float32))

    def __len__(self) -> int:
        return self.configuration.shape[0]

    def __getitem__(self, index: int):
        return self.configuration[index], self.frequency, self.response[index]


def _build_masked_loaders(dataset, freq_indices: np.ndarray, batch_size: int, seed: int) -> dict[str, DataLoader]:
    splits = _split_ids(dataset)
    norm = dataset.norm_params
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders = {}
    for name in ("train", "val", "test"):
        subset = _MaskedFrequencyDataset(dataset, splits[name], freq_indices, norm)
        loaders[name] = DataLoader(
            subset, batch_size=batch_size, shuffle=(name == "train"),
            generator=generator if name == "train" else None,
        )
    return loaders


def _metrics_on(pred: np.ndarray, true: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    p, t = pred[:, idx], true[:, idx]
    err = p - t
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    tc = t - t.mean(axis=1, keepdims=True)
    pc = p - p.mean(axis=1, keepdims=True)
    num = np.sum(tc * pc, axis=1)
    den = np.sqrt(np.sum(tc**2, axis=1) * np.sum(pc**2, axis=1))
    pear = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return rmse, mae, float(np.nanmean(pear))


def run_frequency_holdout(
    operator_specs: list[dict[str, object]] | None = None,
    num_configurations: int = 10000,
    epochs: int = 200,
    batch_size: int = 16,
    dataset_file: str | list[str] = "datasets/dataset_erp_ft.pth",
    holdout_start_frac: float = 0.40,
    holdout_end_frac: float = 0.60,
    num_plot: int = 5,
    seed: int = 727,
    use_sorted_branch: bool = False,
) -> dict[str, dict[str, object]]:
    """Run the frequency-holdout experiment for the given operators (default: all).

    ``holdout_start_frac``/``holdout_end_frac`` locate the held-out
    contiguous band as a fraction of the full frequency range (default: the
    middle 20%, matching the original experiment). Returns a dict keyed by
    operator short name with seen/unseen RMSE/MAE/Pearson + loss history.
    """
    specs = operator_specs if operator_specs is not None else list(OPERATORS.values())
    if not specs:
        raise ValueError("operator_specs must not be empty.")
    if not (0.0 <= holdout_start_frac < holdout_end_frac <= 1.0):
        raise ValueError("Need 0 <= holdout_start_frac < holdout_end_frac <= 1.")

    print("Loading dataset...")
    dataset, _ = prepare_operator_data(
        num_configurations=num_configurations, batch_size=batch_size,
        dataset_file=dataset_file, regenerate_dataset=False, seed=seed,
    )
    n_freq = dataset.frequency_values.shape[0]
    start = int(n_freq * holdout_start_frac)
    end = int(n_freq * holdout_end_frac)
    unseen_idx = np.arange(start, end)
    seen_mask = np.ones(n_freq, dtype=bool)
    seen_mask[start:end] = False
    seen_idx = np.nonzero(seen_mask)[0]
    freq_hz = np.asarray(dataset.frequency_values)
    print(
        f"n_freq={n_freq}, holdout band indices=[{start},{end}) -> {len(unseen_idx)} bins "
        f"({100 * len(unseen_idx) / n_freq:.1f}%), seen={len(seen_idx)} bins"
    )
    print(
        f"Holdout frequency range: {freq_hz[start]:.1f}-{freq_hz[end - 1]:.1f} Hz "
        f"(full range {freq_hz[0]:.1f}-{freq_hz[-1]:.1f} Hz)"
    )

    dataset_tag = dataset_tag_for(dataset_file)
    summary_dir = forward_plot_root(dataset_tag, FREQ_HOLDOUT) / "ALL_MODELS"
    summary_dir.mkdir(parents=True, exist_ok=True)

    seen_loaders = _build_masked_loaders(dataset, seen_idx, batch_size=batch_size, seed=seed)
    full_loaders = _build_masked_loaders(dataset, np.arange(n_freq), batch_size=batch_size, seed=seed)
    train_val_loaders = {"train": seen_loaders["train"], "val": seen_loaders["val"]}
    test_loader_full = full_loaders["test"]

    norm = dataset.norm_params
    results: dict[str, dict[str, object]] = {}

    names = []
    for spec in specs:
        # Optional f_t-sorted resonator branch (set-encoder architectures
        # only), saved/plotted as <MODEL>_sorted.
        sorted_variant = bool(use_sorted_branch and spec.get("supports_sorted_branch"))
        short = f"{spec['short']}_sorted" if sorted_variant else spec["short"]
        model_config = {**dict(spec["model_config"]), **({"use_sorted_branch": True} if sorted_variant else {})}
        names.append(short)
        print("\n" + "#" * 76)
        print(f"Frequency-holdout training: {short}")
        print("#" * 76)
        model = spec["build_model"](num_res=dataset.num_res, **model_config).to(device)
        print(f"{short} trainable parameters: {parameter_count(model):,}")

        model, history = train_operator(
            model, train_val_loaders, epochs=epochs, lr=float(spec["lr"]), lbfgs_epochs=0,
            plot=False, save_plots=False, operator_name=f"{short}_holdout",
        )

        model.eval()
        preds, trues, configs = [], [], []
        with torch.inference_mode():
            for configuration, frequency, target in test_loader_full:
                configs.append(configuration.numpy())
                configuration = configuration.to(device, dtype=torch.float32)
                frequency = frequency.to(device, dtype=torch.float32)
                prediction = model(configuration, frequency)
                preds.append(prediction.cpu().numpy())
                trues.append(target.numpy())
        pred = denormalize_erp_array(np.concatenate(preds, axis=0)[..., 0], norm)
        true = denormalize_erp_array(np.concatenate(trues, axis=0)[..., 0], norm)
        physical_configs = denormalize_configuration_array(np.concatenate(configs, axis=0), norm)

        rmse_seen, mae_seen, pear_seen = _metrics_on(pred, true, seen_idx)
        rmse_unseen, mae_unseen, pear_unseen = _metrics_on(pred, true, unseen_idx)
        print(f"{short}: SEEN   RMSE={rmse_seen:.4f} dB  MAE={mae_seen:.4f} dB  Pearson={pear_seen:.4f}")
        print(f"{short}: UNSEEN RMSE={rmse_unseen:.4f} dB  MAE={mae_unseen:.4f} dB  Pearson={pear_unseen:.4f}")
        print(f"{short}: generalization gap (unseen-seen) RMSE = {rmse_unseen - rmse_seen:+.4f} dB")

        results[short] = dict(
            rmse_seen=rmse_seen, mae_seen=mae_seen, pearson_seen=pear_seen,
            rmse_unseen=rmse_unseen, mae_unseen=mae_unseen, pearson_unseen=pear_unseen,
            history_train=[float(v) for v in history["train"]],
            history_val=[float(v) for v in history["val"]],
        )

        # erp_forward_operators/plots/FREQ_HOLDOUT/<dataset>/<MODEL>/
        plot_dir = forward_plot_dir(short, dataset_tag, FREQ_HOLDOUT)

        plot_loss_curves(
            history["train"], history["val"],
            title=f"{short}: frequency-holdout training history",
            ylabel="Normalised ERP loss (seen frequencies)", log_y=True,
            save_path=plot_dir / "loss_curve.png", show=False,
        )
        for i in range(min(num_plot, pred.shape[0])):
            plot_erp_comparison(
                freq_hz, true[i], pred[i],
                title=f"{short}: test configuration {i + 1} (frequency holdout)",
                configuration=physical_configs[i],
                highlight_band=(freq_hz[start], freq_hz[end - 1]),
                save_path=plot_dir / f"erp_comparison_config_{i + 1:02d}.png", show=False,
            )
        plot_prediction_scatter(
            true[:, unseen_idx], pred[:, unseen_idx],
            xlabel=f"True {ERP_LABEL}", ylabel=f"Predicted {ERP_LABEL}",
            title=f"{short}: held-out band only (never trained on)",
            save_path=plot_dir / "prediction_vs_ground_truth_unseen_band.png", show=False,
        )
        plot_prediction_scatter(
            true[:, seen_idx], pred[:, seen_idx],
            xlabel=f"True {ERP_LABEL}", ylabel=f"Predicted {ERP_LABEL}",
            title=f"{short}: seen frequencies",
            save_path=plot_dir / "prediction_vs_ground_truth_seen.png", show=False,
        )

        checkpoint_path = forward_model_path(short, dataset_tag, FREQ_HOLDOUT)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "operator_name": short,
                "model_config": model_config,
                "model_state_dict": model.state_dict(),
                "preprocessing_state": dataset.preprocessing_state(),
                "modal_resolution": list(dataset.modal_resolution),
                "holdout_band_indices": [int(start), int(end)],
                "training_config": {"epochs": int(epochs), "batch_size": int(batch_size),
                                    "learning_rate": float(spec["lr"]), "optimizer": "AdamW", "dataset_tag": dataset_tag},
                "history": {"train": results[short]["history_train"], "val": results[short]["history_val"]},
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint {checkpoint_path} + plots in {plot_dir}")

    plot_all_models_loss(
        {m: {"train": r["history_train"], "val": r["history_val"]} for m, r in results.items()},
        title="Frequency-holdout training history: all operators",
        ylabel="Normalised ERP loss",
        save_path=summary_dir / "all_models_loss.png",
    )

    with open(summary_dir / "results.json", "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk not in ("history_train", "history_val")}
             for k, v in results.items()},
            f, indent=2,
        )

    models_order = names
    rmse_seen_vals = [results[m]["rmse_seen"] for m in models_order]
    rmse_unseen_vals = [results[m]["rmse_unseen"] for m in models_order]
    x = np.arange(len(models_order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width / 2, rmse_seen_vals, width, label="Seen frequencies (trained on)")
    b2 = ax.bar(x + width / 2, rmse_unseen_vals, width, label="Held-out band (never trained on)")
    ax.set_xticks(x)
    ax.set_xticklabels(models_order)
    ax.set_ylabel("RMSE (dB)")
    ax.set_title("Frequency-holdout generalisation: seen vs unseen RMSE")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    fig.tight_layout()
    save_figure(fig, summary_dir / "seen_vs_unseen_rmse.png")

    print("\nFINAL SUMMARY")
    header = (
        f"{'Model':6s} | {'RMSE seen':>10s} | {'RMSE unseen':>12s} | {'Gap':>8s} | "
        f"{'Pearson seen':>13s} | {'Pearson unseen':>15s}"
    )
    print(header)
    for m in models_order:
        r = results[m]
        print(
            f"{m:6s} | {r['rmse_seen']:10.4f} | {r['rmse_unseen']:12.4f} | "
            f"{r['rmse_unseen'] - r['rmse_seen']:+8.4f} | {r['pearson_seen']:13.4f} | "
            f"{r['pearson_unseen']:15.4f}"
        )
    print("DONE")
    return results
