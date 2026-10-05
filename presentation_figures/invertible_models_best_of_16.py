"""Best of 16 sampled designs of iFNO, iLNO, iSTO and iGNO (test configuration 3), all in one plot: the ERPs of the best designs, checked with the
solver, against the target ERP; the plate with the true resonators and the best designs; table of the predicted mass and tuning frequency.
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

for r in rows:
    r["design"] = r["samples"][r["best"]]
    r["curve"] = r["solved"][r["best"]]

fig = plt.figure(figsize=(17.5, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.25, 1.0], wspace=0.18, hspace=0.34)
ax = fig.add_subplot(gs[:, 0])
ax.plot(freq, target, color="black", lw=2.8, label="Target ERP")
for r in rows:
    ax.plot(freq, r["curve"], color=COLOURS[r["label"]], lw=1.6, ls="--", label=f"{r['label']} (RMSE {r['rmse_best']:.2f} dB)")
for j, ft in enumerate(true_design[:, 2]):
    ax.axvline(ft, color="#c2412c", ls="--", lw=1.3, zorder=1, label="True tuning frequencies" if j == 0 else None)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("ERP (dB)")
ax.grid(True, color="#d9d9d4", lw=0.5)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))

axp = fig.add_subplot(gs[0, 1])
axp.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, ec="black", lw=1.6))
axp.plot([xf], [yf], marker="*", color="#35d0ff", ms=18, mec="black", ls="", zorder=5)
for r in rows:
    c = COLOURS[r["label"]]
    for i in range(num_res):
        axp.plot([true_design[i, 3], r["design"][i, 3]], [true_design[i, 4], r["design"][i, 4]], color=c, lw=0.9, ls=":", zorder=3)
    axp.plot(r["design"][:, 3], r["design"][:, 4], "X", color=c, ms=10, mec="black", mew=0.6, ls="", zorder=6)
axp.plot(true_design[:, 3], true_design[:, 4], "o", color="#dc143c", ms=12, mec="black", ls="", zorder=7)
for i in range(num_res):
    axp.annotate(f"R{i + 1}", (true_design[i, 3], true_design[i, 4]), xytext=(-4, 9), textcoords="offset points", fontsize=11, ha="right", zorder=8)
axp.set_xlim(-0.05, Lx + 0.05)
axp.set_ylim(-0.05, Ly + 0.05)
axp.set_aspect("equal")
axp.set_xlabel("Position $x$ (m)")
axp.set_ylabel("Position $y$ (m)")
axp.set_title("Plate configuration: true resonators (circles) and best-of-16 designs (crosses)", fontsize=11)
handles = [Line2D([], [], marker="*", color="#35d0ff", mec="black", ls="", ms=13, label="Force"),
           Line2D([], [], marker="o", color="#dc143c", mec="black", ls="", ms=9, label="True resonator")]
handles += [Line2D([], [], marker="X", color=COLOURS[r["label"]], mec="black", mew=0.6, ls="", ms=9, label=r["label"]) for r in rows]
axp.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=6, frameon=False, fontsize=9, handletextpad=0.2, columnspacing=1.0)

axt = fig.add_subplot(gs[1, 1])
axt.axis("off")
cells = [["True"] + [f"{true_design[i, 0]:.2f} kg, {true_design[i, 2]:.0f} Hz" for i in range(num_res)]]
for r in rows:
    cells.append([r["label"]] + [f"{r['design'][i, 0]:.2f} kg, {r['design'][i, 2]:.0f} Hz" for i in range(num_res)])
table = axt.table(cellText=cells, colLabels=[""] + [f"R{i + 1}: mass, tuning frequency" for i in range(num_res)], loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11.5)
table.scale(1.0, 1.7)
for (ri, ci), cell in table.get_celld().items():
    cell.set_edgecolor("#c3ccd6")
    if ri == 0:
        cell.set_facecolor("#eef2f7")
    elif ci == 0:
        cell.set_text_props(color=(["black"] + [COLOURS[r["label"]] for r in rows])[ri - 1])
fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.1)
for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"invertible_models_best_of_16.{ext}", facecolor="white", **kw)
print("saved")
