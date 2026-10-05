"""Forward ERP prediction of the invertible operators (erp_invertible/models/200k_2res_18modes) on one test configuration, all in one plot.

Each checkpoint is loaded as in erp_invertible/scripts/train.py (_load_checkpoint / evaluate_one); the configuration is the TEST_INDEX-th
test configuration of the 200k_2res_18modes split (index 3, as in the iDON inverse example). Ground truth: the test spectrum (solver).
"""
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
from matplotlib.patches import Rectangle
from utils.physics import Lx, Ly, xf, yf
from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_invertible.scripts.registry import INVERTIBLE_OPERATORS
from erp_invertible.scripts.train import _design_dim_of
from utils.erp_dataset import DATASETS, denormalize_erp_array, select_dataset_modal_resolution
from utils.paths import invertible_model_path
from utils.support import device

TAG = "200k_2res_18modes"
TEST_INDEX = 3  # 1-based test configuration
MODELS = [  # label, short name in the registry, checkpoint name, colour
    ("iFNO", "iFNO", "ifno_sortenc_phys_b12_bg_bin_pad_cyc_s2e", "#d62728"),
    ("iLNO", "iLNO", "ilno_sortenc_phys_b12_bg_bin_cyc_s2e", "#2ca02c"),
    ("iSTO", "iSTO", "isto_phys_b12_bg_bin_cyc_s2e", "#1f77b4"),
    ("iGNO", "iGNO", "igno_phys_b12_bg_bin_cyc_s2e", "#ff7f0e"),
]
spec_by_short = {s["short"]: s for s in INVERTIBLE_OPERATORS.values()}

select_dataset_modal_resolution(TAG)
files = list(DATASETS[TAG]["files"])
dataset, loaders = prepare_inverse_data(num_configurations=int(DATASETS[TAG]["num_configurations"]), batch_size=128,
                                        dataset_file=files if len(files) > 1 else files[0], seed=727, design_param="bounded12")
spectrum, design = next(iter(loaders["test"]))
idx = TEST_INDEX - 1
truth = denormalize_erp_array(spectrum[idx : idx + 1], dataset.norm_params)[0]
freq = np.asarray(dataset.frequency_values)
config = denormalize_design(design[idx : idx + 1].reshape(1, -1).numpy(), int(dataset.norm_params["num_res"]),
                            dataset.norm_params, consistent=False)[0]  # (R, 5): m, k, f_t, x, y
print("configuration [m, k, f_t, x, y]:\n", np.round(config, 3))

curves, designs = {}, {}
num_res = int(dataset.norm_params["num_res"])
for label, short, name, colour in MODELS:
    ck = torch.load(invertible_model_path(name, TAG), map_location=device, weights_only=False)
    cfg = dict(ck["model_config"])
    spec = spec_by_short[short]
    model = spec["build"](design_dim=_design_dim_of(cfg), **cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    with torch.no_grad():
        pred = model.predict_spectrum_from_design(design[idx : idx + 1].to(device).reshape(1, -1)).squeeze(-1).cpu()
    with torch.no_grad():  # inverse: design estimated from the target ERP (point estimate), sorted by f_t like the truth
        point = model.infer_point_estimate(spectrum[idx : idx + 1].to(device)).cpu()
    designs[label] = sort_resonators_by_ft(denormalize_design(point.numpy(), num_res, dataset.norm_params))[0]
    curves[label] = denormalize_erp_array(pred, dataset.norm_params)[0]
    print(f"{label}: MSE {((curves[label] - truth) ** 2).mean():.3f} dB^2, MAE {np.abs(curves[label] - truth).mean():.3f} dB")

fig = plt.figure(figsize=(17.5, 6.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.25, 1.0], wspace=0.18, hspace=0.34)
ax = fig.add_subplot(gs[:, 0])
ax.plot(freq, truth, color="black", lw=2.8, label="Ground truth (solver)")
for (label, _, _, colour) in MODELS:
    ax.plot(freq, curves[label], color=colour, lw=1.6, ls="--", label=label)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("ERP (dB)")
ax.grid(True, color="#d9d9d4", lw=0.5)
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07))

axp = fig.add_subplot(gs[0, 1])
axp.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, ec="black", lw=1.6))
axp.plot([xf], [yf], marker="*", color="#35d0ff", ms=18, mec="black", ls="", zorder=5)
for (label, _, _, colour) in MODELS:
    for r in range(num_res):
        axp.plot([config[r, 3], designs[label][r, 3]], [config[r, 4], designs[label][r, 4]], color=colour, lw=0.9, ls=":", zorder=3)
    axp.plot(designs[label][:, 3], designs[label][:, 4], "X", color=colour, ms=10, mec="black", mew=0.6, ls="", zorder=6)
axp.plot(config[:, 3], config[:, 4], "o", color="#dc143c", ms=12, mec="black", ls="", zorder=7)
for i, (m, k, ft, x, y) in enumerate(config, 1):
    axp.annotate(f"R{i}", (x, y), xytext=(-4, 9), textcoords="offset points", fontsize=11, ha="right", zorder=8)
axp.set_xlim(-0.05, Lx + 0.05)
axp.set_ylim(-0.05, Ly + 0.05)
axp.set_aspect("equal")
axp.set_xlabel("Position $x$ (m)")
axp.set_ylabel("Position $y$ (m)")
axp.set_title("Plate configuration: true resonators (circles) and predicted designs (crosses)", fontsize=11)
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="*", color="#35d0ff", mec="black", ls="", ms=13, label="Force"),
           Line2D([], [], marker="o", color="#dc143c", mec="black", ls="", ms=9, label="True resonator")]
handles += [Line2D([], [], marker="X", color=c, mec="black", mew=0.6, ls="", ms=9, label=l) for l, _, _, c in MODELS]
axp.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=6, frameon=False, fontsize=9, handletextpad=0.2, columnspacing=1.0)

axt = fig.add_subplot(gs[1, 1])
axt.axis("off")
rows = [["True"] + [f"{config[r, 0]:.2f} kg, {config[r, 2]:.0f} Hz" for r in range(num_res)]]
for (label, _, _, _) in MODELS:
    rows.append([label] + [f"{designs[label][r, 0]:.2f} kg, {designs[label][r, 2]:.0f} Hz" for r in range(num_res)])
table = axt.table(cellText=rows, colLabels=[""] + [f"R{r + 1}: mass, tuning frequency" for r in range(num_res)], loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11.5)
table.scale(1.0, 1.7)
for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#c3ccd6")
    if r == 0:
        cell.set_facecolor("#eef2f7")
    elif c == 0 and r >= 1:
        cell.set_text_props(color=(["black"] + [m[3] for m in MODELS])[r - 1])
fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.1)
for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"invertible_models_one_plot.{ext}", facecolor="white", **kw)
print("saved")
