"""DCO trained with the peak term from the first epoch (erp_forward/models/100k/dco.pth) versus from 80 % of the
epochs (erp_forward/models/experiments/dco_variants/dco_staged.pth), on the 100k dataset's test split.

Both checkpoints were identified from the repo's peak_term_evolution plots: on the first test configuration they give
the plot's errors (12.168 and 15.719 dB^2). Output: the first N_CONFIGS test configurations as spectra, and metrics over
the whole test split (RMSE, mean absolute error at every true resonance peak, at the tallest peak).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.plot_style  # noqa: F401
from erp_forward.scripts import dco
from erp_forward.scripts.neural_operator_utils import (build_operator_model, load_operator_checkpoint, prepare_operator_data,
                                                       spectrum_peak_mask)
from utils.erp_dataset import denormalize_configuration_array, denormalize_erp_array
from utils.support import device

N_CONFIGS = 8
FILES = ["datasets/erp/3res/100k/dataset_erp_ft_100k_part1.pth", "datasets/erp/3res/100k/dataset_erp_ft_100k_part2.pth"]
MODELS = {  # label -> (checkpoint, colour)
    "peak term from the first epoch": ("erp_forward/models/100k/dco.pth", "#d62728"),
    "peak term from 80% of the epochs": ("erp_forward/models/experiments/dco_variants/dco_staged.pth", "#2ca02c"),
}
OUT = ROOT / "presentation_figures"


def predict(checkpoint):
    ck = load_operator_checkpoint(checkpoint)
    ps = dict(ck["preprocessing_state"])
    ps["dataset_file"] = FILES
    ids = np.asarray(ps["selected_source_ids"])
    ds, loaders = prepare_operator_data(num_configurations=int(ids.size), batch_size=256, dataset_file=FILES,
                                        preprocessing_state=ps, verbose=False)
    model = build_operator_model(dco.build_model, ds.num_res, ck["model_config"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    preds, trues, cfgs = [], [], []
    with torch.inference_mode():
        for cfg, fr, tg in loaders["test"]:
            out = model(cfg.to(device, dtype=torch.float32), fr.to(device, dtype=torch.float32))
            preds.append(out.cpu().numpy()[..., 0]); trues.append(tg.numpy()[..., 0]); cfgs.append(cfg.numpy())
    pred = denormalize_erp_array(np.concatenate(preds), ds.norm_params)
    true = denormalize_erp_array(np.concatenate(trues), ds.norm_params)
    cfg = denormalize_configuration_array(np.concatenate(cfgs), ds.norm_params)
    return pred, true, cfg, np.asarray(ds.frequency_values)


def metrics(pred, true):
    err = pred - true
    mask = spectrum_peak_mask(torch.from_numpy(true)[..., None].float()).numpy()[..., 0]
    top = true.argmax(axis=1)
    return {"rmse_db": float(np.sqrt((err ** 2).mean())), "mae_db": float(np.abs(err).mean()),
            "peak_mae_db": float(np.abs(err)[mask].mean()),
            "top_peak_mae_db": float(np.abs(err[np.arange(len(err)), top]).mean()), "num_test": int(len(err))}


results = {label: predict(path) for label, (path, _) in MODELS.items()}
true, cfg, freq = next(iter(results.values()))[1:]
stats = {label: metrics(p, true) for label, (p, *_) in results.items()}
(OUT / "peak_schedule_metrics.json").write_text(json.dumps(stats, indent=2))
for label, m in stats.items():
    print(f"{label}: RMSE {m['rmse_db']:.2f} | MAE {m['mae_db']:.2f} | peak MAE {m['peak_mae_db']:.2f} | tallest-peak MAE {m['top_peak_mae_db']:.2f} dB ({m['num_test']} test spectra)")

fig, axes = plt.subplots(N_CONFIGS // 2, 2, figsize=(15, 3.3 * N_CONFIGS // 2), sharex=True)
for i, ax in enumerate(axes.ravel()):
    ax.plot(freq, true[i], color="black", lw=2.6, label="Ground truth")
    for label, (path, colour) in MODELS.items():
        mse = float(((results[label][0][i] - true[i]) ** 2).mean())
        ax.plot(freq, results[label][0][i], color=colour, lw=1.6, ls="--", label=f"{label}" if i == 0 else None)
        ax.text(0.99, 0.06 + 0.12 * list(MODELS).index(label), rf"MSE $={mse:.1f}$", transform=ax.transAxes, ha="right",
                color=colour, fontsize=10)
    for ft in cfg[i][:, 2]:
        ax.axvline(ft, color="#c2412c", ls=":", lw=0.9)
    ax.set_title(f"Test configuration {i + 1}", fontsize=11)
    ax.grid(True, color="#d9d9d4", lw=0.5)
    ax.set_ylabel("ERP (dB)")
for ax in axes[-1]:
    ax.set_xlabel("Frequency (Hz)")
fig.legend(*axes.ravel()[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=[0, 0, 1, 0.965])
for ext, kw in (("png", dict(dpi=170)), ("pdf", {})):
    fig.savefig(OUT / f"peak_schedule_comparison.{ext}", facecolor="white", **kw)
print("saved")
