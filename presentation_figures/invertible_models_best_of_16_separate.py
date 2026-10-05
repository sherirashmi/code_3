"""Best of 16 sampled designs of iFNO, iLNO, iSTO and iGNO (test configuration 3): one ERP graph per model (best design checked with the
solver against the target ERP) and one plate with the true resonators and the best designs of all four models.
"best of 16": of 16 designs sampled through the VAE, the one whose solver ERP is closest to the target (needs the solver and the target).
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

COLOURS = {"iFNO": "#d62728", "iLNO": "#2ca02c", "iSTO": "#1f77b4", "iGNO": "#ff7f0e"}
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from utils.physics import Lx, Ly, xf, yf

OUT = ROOT / "presentation_figures"
for r in rows:
    r["design"] = r["samples"][r["best"]]
    r["curve"] = r["solved"][r["best"]]
lo = min(float(target.min()), *(float(r["curve"].min()) for r in rows)) - 3
hi = max(float(target.max()), *(float(r["curve"].max()) for r in rows)) + 3


def save(fig, name):
    for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", name)


for r in rows:  # one ERP graph per model
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(freq, target, color="black", lw=2.8, label="Target ERP")
    ax.plot(freq, r["curve"], color=COLOURS[r["label"]], lw=1.9, ls="--", label=f"{r['label']}, best of {NUM_SAMPLES} (RMSE {r['rmse_best']:.2f} dB)")
    for j, ft in enumerate(true_design[:, 2]):
        ax.axvline(ft, color="#c2412c", ls="--", lw=1.2, zorder=1, label="True tuning frequencies" if j == 0 else None)
    ax.set_xlim(freq[0], freq[-1])
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ERP (dB)")
    ax.set_title(r["label"], fontsize=14)
    ax.grid(True, color="#d9d9d4", lw=0.5)
    ax.legend(frameon=False, loc="lower right", fontsize=10)
    fig.tight_layout()
    save(fig, f"invertible_best16_{r['label']}")

fig, axp = plt.subplots(figsize=(9.5, 4.8))  # one plate for all models
axp.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, ec="black", lw=1.6))
axp.plot([xf], [yf], marker="*", color="#35d0ff", ms=18, mec="black", ls="", zorder=5)
for r in rows:
    c = COLOURS[r["label"]]
    for i in range(num_res):
        axp.plot([true_design[i, 3], r["design"][i, 3]], [true_design[i, 4], r["design"][i, 4]], color=c, lw=0.9, ls=":", zorder=3)
    axp.plot(r["design"][:, 3], r["design"][:, 4], "X", color=c, ms=11, mec="black", mew=0.6, ls="", zorder=6)
axp.plot(true_design[:, 3], true_design[:, 4], "o", color="#dc143c", ms=13, mec="black", ls="", zorder=7)
for i in range(num_res):
    axp.annotate(f"R{i + 1}", (true_design[i, 3], true_design[i, 4]), xytext=(-4, 9), textcoords="offset points", fontsize=12, ha="right", zorder=8)
axp.set_xlim(-0.05, Lx + 0.05)
axp.set_ylim(-0.05, Ly + 0.05)
axp.set_aspect("equal")
axp.set_xlabel("Position $x$ (m)")
axp.set_ylabel("Position $y$ (m)")
handles = [Line2D([], [], marker="*", color="#35d0ff", mec="black", ls="", ms=14, label="Force"),
           Line2D([], [], marker="o", color="#dc143c", mec="black", ls="", ms=10, label="True resonator")]
handles += [Line2D([], [], marker="X", color=COLOURS[r["label"]], mec="black", mew=0.6, ls="", ms=10, label=r["label"]) for r in rows]
axp.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=6, frameon=False, fontsize=10, handletextpad=0.2, columnspacing=1.0)
fig.tight_layout()
save(fig, "invertible_best16_plate")
