import sys, os, json, argparse
sys.path.insert(0, "/home/user/code_3")
os.chdir("/home/user/code_3")

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from forward_operators.neural_operator_utils import (
    prepare_operator_data,
    _split_ids,
    _configuration_features,
    train_operator,
    parameter_count,
    device,
)
from utils.erp_dataset import (
    normalize_configuration_array,
    normalize_frequency_array,
    normalize_erp_array,
    denormalize_erp_array,
)
from forward_operators.operator_registry import OPERATORS

OP_KEYS = {"DON": "1", "DNO": "2", "DCO": "4", "WNO": "8", "GNO": "5", "NN": "9"}


class MaskedFrequencyDataset(Dataset):
    def __init__(self, dataset, configuration_ids, freq_indices, norm):
        ids = np.asarray(configuration_ids, dtype=np.int64)
        configuration = normalize_configuration_array(_configuration_features(dataset)[ids], norm)
        frequency_full = normalize_frequency_array(dataset.frequency_values, norm)
        response_full = normalize_erp_array(
            np.asarray(dataset.responses, dtype=np.float32)[ids, :, 0], norm
        )
        self.configuration = torch.from_numpy(configuration.astype(np.float32))
        self.frequency = torch.from_numpy(frequency_full[freq_indices][:, None].astype(np.float32))
        self.response = torch.from_numpy(
            response_full[:, freq_indices][..., None].astype(np.float32)
        )

    def __len__(self):
        return self.configuration.shape[0]

    def __getitem__(self, index):
        return self.configuration[index], self.frequency, self.response[index]


def build_masked_loaders(dataset, freq_indices, batch_size=16, seed=727):
    splits = _split_ids(dataset)
    norm = dataset.norm_params
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders = {}
    for name in ("train", "val", "test"):
        subset = MaskedFrequencyDataset(dataset, splits[name], freq_indices, norm)
        loaders[name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            generator=generator if name == "train" else None,
        )
    return loaders


def metrics_on(pred, true, idx):
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


def main(num_configurations=10000, epochs=200, num_plot=5):
    print("Loading dataset...")
    dataset, _ = prepare_operator_data(
        num_configurations=num_configurations,
        batch_size=16,
        dataset_file="datasets/dataset_erp_ft.pth",
        regenerate_dataset=False,
        seed=727,
    )
    n_freq = dataset.frequency_values.shape[0]
    start = int(n_freq * 0.40)
    end = int(n_freq * 0.60)
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
        f"Holdout frequency range: {freq_hz[start]:.1f} - {freq_hz[end - 1]:.1f} Hz "
        f"(full range {freq_hz[0]:.1f} - {freq_hz[-1]:.1f} Hz)"
    )

    seen_loaders = build_masked_loaders(dataset, seen_idx, batch_size=16, seed=727)
    full_loaders = build_masked_loaders(dataset, np.arange(n_freq), batch_size=16, seed=727)
    train_val_loaders = {"train": seen_loaders["train"], "val": seen_loaders["val"]}
    test_loader_full = full_loaders["test"]

    norm = dataset.norm_params
    results = {}
    out_root = Path("plots/FREQ_HOLDOUT")
    out_root.mkdir(parents=True, exist_ok=True)

    for short, key in OP_KEYS.items():
        print("\n" + "#" * 76)
        print(f"Frequency-holdout training: {short}")
        print("#" * 76)
        spec = OPERATORS[key]
        model = spec["build_model"](num_res=dataset.num_res, **spec["model_config"]).to(device)
        print(f"{short} trainable parameters: {parameter_count(model):,}")

        model, history = train_operator(
            model,
            train_val_loaders,
            epochs=epochs,
            lr=5e-4,
            weight_decay=1e-4,
            slope_weight=0.5,
            lbfgs_epochs=0,
            plot=False,
            save_plots=False,
            operator_name=f"{short}_holdout",
        )

        model.eval()
        preds, trues, configs = [], [], []
        with torch.inference_mode():
            for configuration, frequency, target in test_loader_full:
                configuration = configuration.to(device, dtype=torch.float32)
                frequency = frequency.to(device, dtype=torch.float32)
                prediction = model(configuration, frequency)
                preds.append(prediction.cpu().numpy())
                trues.append(target.numpy())
                configs.append(configuration.cpu().numpy())
        pred = denormalize_erp_array(np.concatenate(preds, axis=0)[..., 0], norm)
        true = denormalize_erp_array(np.concatenate(trues, axis=0)[..., 0], norm)

        rmse_seen, mae_seen, pear_seen = metrics_on(pred, true, seen_idx)
        rmse_unseen, mae_unseen, pear_unseen = metrics_on(pred, true, unseen_idx)
        print(f"{short}: SEEN   RMSE={rmse_seen:.4f} dB  MAE={mae_seen:.4f} dB  Pearson={pear_seen:.4f}")
        print(f"{short}: UNSEEN RMSE={rmse_unseen:.4f} dB  MAE={mae_unseen:.4f} dB  Pearson={pear_unseen:.4f}")
        print(f"{short}: generalization gap (unseen-seen) RMSE = {rmse_unseen - rmse_seen:+.4f} dB")

        results[short] = dict(
            rmse_seen=rmse_seen, mae_seen=mae_seen, pearson_seen=pear_seen,
            rmse_unseen=rmse_unseen, mae_unseen=mae_unseen, pearson_unseen=pear_unseen,
            history_train=[float(v) for v in history["train"]],
            history_val=[float(v) for v in history["val"]],
        )

        plot_dir = out_root / short
        plot_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(num_plot, pred.shape[0])):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(freq_hz, true[i], lw=2.5, label="Ground truth")
            ax.plot(freq_hz, pred[i], "--", lw=2, label="Prediction")
            ax.axvspan(
                freq_hz[start], freq_hz[end - 1], color="orange", alpha=0.15,
                label="Held-out band (never trained on)",
            )
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("ERP (dB)")
            ax.set_title(f"{short} - test config {i + 1:02d} - frequency-holdout generalization")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / f"holdout_spectrum_config_{i + 1:02d}.png", dpi=140)
            plt.close(fig)

        torch.save(model.state_dict(), f"models/{short.lower()}_freq_holdout.pth")
        print(f"Saved checkpoint + plots for {short}")

    with open(out_root / "results.json", "w") as f:
        json.dump(
            {k: {kk: vv for kk, vv in v.items() if kk not in ("history_train", "history_val")}
             for k, v in results.items()},
            f, indent=2,
        )

    models_order = list(OP_KEYS.keys())
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
    ax.set_title("Frequency-holdout generalization: seen vs unseen RMSE per operator")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "seen_vs_unseen_rmse.png", dpi=150)
    plt.close(fig)
    print("Saved comparison chart")

    print("\nFINAL SUMMARY")
    header = f"{'Model':6s} | {'RMSE seen':>10s} | {'RMSE unseen':>12s} | {'Gap':>8s} | {'Pearson seen':>13s} | {'Pearson unseen':>15s}"
    print(header)
    for m in models_order:
        r = results[m]
        print(
            f"{m:6s} | {r['rmse_seen']:10.4f} | {r['rmse_unseen']:12.4f} | "
            f"{r['rmse_unseen'] - r['rmse_seen']:+8.4f} | {r['pearson_seen']:13.4f} | "
            f"{r['pearson_unseen']:15.4f}"
        )
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-configurations", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    main(num_configurations=args.num_configurations, epochs=args.epochs)
