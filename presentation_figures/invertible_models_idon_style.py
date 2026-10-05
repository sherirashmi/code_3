"""Inverse design of test configuration 3 by iFNO, iLNO, iSTO and iGNO, in the layout of the Invertible DeepONet example:
left, the ERP of the predicted designs checked with the solver (16 designs sampled through the VAE, the point estimate, the best
of the 16) against the target ERP; right, the plate with the true resonators and the predicted designs.
Models and test configuration as in invertible_models_one_plot.py.
"""
import importlib
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
from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_invertible.scripts.registry import INVERTIBLE_OPERATORS
from erp_invertible.scripts.train import _design_dim_of
from utils.erp_dataset import DATASETS, configuration_to_resonators, denormalize_erp_array, select_dataset_modal_resolution
from utils.paths import invertible_model_path
from utils.solver import compute_erp_spectrum
from utils.support import device

_draw_plate = importlib.import_module("erp_inverse.scripts.fixed_resonator.evaluate")._draw_plate

TAG = "200k_2res_18modes"
TEST_INDEX = 3
NUM_SAMPLES = 16
MODELS = [
    ("iFNO", "ifno_sortenc_phys_b12_bg_bin_pad_cyc_s2e"),
    ("iLNO", "ilno_sortenc_phys_b12_bg_bin_cyc_s2e"),
    ("iSTO", "isto_phys_b12_bg_bin_cyc_s2e"),
    ("iGNO", "igno_phys_b12_bg_bin_cyc_s2e"),
]
spec_by_short = {s["short"]: s for s in INVERTIBLE_OPERATORS.values()}

select_dataset_modal_resolution(TAG)
files = list(DATASETS[TAG]["files"])
dataset, loaders = prepare_inverse_data(num_configurations=int(DATASETS[TAG]["num_configurations"]), batch_size=128,
                                        dataset_file=files if len(files) > 1 else files[0], seed=727, design_param="bounded12")
norm = dataset.norm_params
num_res = int(norm["num_res"])
freq = np.asarray(dataset.frequency_values)
spectrum, design = next(iter(loaders["test"]))
idx = TEST_INDEX - 1
target = denormalize_erp_array(spectrum[idx : idx + 1], norm)[0]
true_design = denormalize_design(design[idx : idx + 1].reshape(1, -1).numpy(), num_res, norm, consistent=False)[0]
spec_t = spectrum[idx : idx + 1].to(device)


def solve(cfg):
    return compute_erp_spectrum(configuration_to_resonators(cfg), frequencies=freq)


rows = []
for label, name in MODELS:
    ck = torch.load(invertible_model_path(name, TAG), map_location=device, weights_only=False)
    cfg = dict(ck["model_config"])
    model = spec_by_short[label]["build"](design_dim=_design_dim_of(cfg), **cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    torch.manual_seed(727)
    with torch.no_grad():
        samples = model.sample(spec_t, num_samples=NUM_SAMPLES).cpu()
        point = model.infer_point_estimate(spec_t).cpu()
    samples = sort_resonators_by_ft(denormalize_design(samples.numpy(), num_res, norm)[0])  # (S, R, 5)
    point = sort_resonators_by_ft(denormalize_design(point.numpy(), num_res, norm))[0]  # (R, 5)
    solved = np.stack([solve(c) for c in samples])
    solved_point = solve(point)
    rmse = np.sqrt(((solved - target) ** 2).mean(axis=1))
    best = int(rmse.argmin())
    rows.append(dict(label=label, samples=samples, point=point, solved=solved, solved_point=solved_point,
                     rmse_point=float(np.sqrt(((solved_point - target) ** 2).mean())), best=best, rmse_best=float(rmse[best])))
    print(f"{label}: point estimate RMSE {rows[-1]['rmse_point']:.2f} dB, best of {NUM_SAMPLES} {rows[-1]['rmse_best']:.2f} dB")

fig, axes = plt.subplots(len(rows), 2, figsize=(15, 5.0 * len(rows)), gridspec_kw={"width_ratios": [1.0, 1.05]})
for (ax, axp), r in zip(axes, rows):
    for s in range(NUM_SAMPLES):
        ax.plot(freq, r["solved"][s], color="0.8", lw=0.6, label=f"All {NUM_SAMPLES} sampled designs" if s == 0 else None)
    ax.plot(freq, target, color="black", lw=2.0, label="Target ERP")
    ax.plot(freq, r["solved_point"], color="#2a78d6", lw=1.5, label=f"Point estimate (RMSE {r['rmse_point']:.2f} dB)")
    ax.plot(freq, r["solved"][r["best"]], color="#eb6834", lw=1.3, ls="--", label=f"Best of {NUM_SAMPLES} (RMSE {r['rmse_best']:.2f} dB)")
    for j, ft in enumerate(true_design[:, 2]):
        ax.axvline(ft, color="#c2412c", ls="--", lw=1.1, zorder=1, label="True tuning frequencies" if j == 0 else None)
    ax.set_xlabel("Frequency $f$ (Hz)")
    ax.set_ylabel("ERP (dB re 1 pW)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title(f"{r['label']}: ERP of the designs (checked with the solver)")
    _draw_plate(axp, true_design, samples=r["samples"], picked=r["point"], picked_label="Point estimate")
    axp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
    tx = "; ".join(f"R{i + 1}: $m$ {true_design[i, 0]:.2f}/{r['point'][i, 0]:.2f} kg, $f_t$ {true_design[i, 2]:.0f}/{r['point'][i, 2]:.0f} Hz"
                   for i in range(num_res))
    axp.set_title(f"Positions (filled: true)\ntrue/predicted: {tx}", fontsize=9)
fig.tight_layout()
for ext, kw in (("png", dict(dpi=170)), ("pdf", {})):
    fig.savefig(ROOT / "presentation_figures" / f"invertible_models_idon_style.{ext}", facecolor="white", **kw)
print("saved")
