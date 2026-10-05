"""DCO on test configuration 4 of the 100k split: no peak term (legacy/dco.pth), peak term from the first epoch
(100k/dco.pth) and staged peak term (100k/dco_sorted_phys.pth, 200 epochs; also sorted resonators and physical features).
Ground truth: the coupled plate-resonator solver. Prediction: each checkpoint through predict_erp_spectrum.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import utils.plot_style  # noqa: F401
from erp_forward.scripts import dco
from erp_forward.scripts.neural_operator_utils import (_configuration_features, _split_ids, build_operator_model, load_operator_checkpoint,
                                                       predict_erp_spectrum, prepare_operator_data)
from utils.physics import Lx, Ly, xf, yf
from utils.support import device

CONFIG = 4  # 1-based test configuration
FILES = ["datasets/erp/3res/100k/dataset_erp_ft_100k_part1.pth", "datasets/erp/3res/100k/dataset_erp_ft_100k_part2.pth"]
MODELS = [  # title, checkpoint, colour
    ("No peak term", "erp_forward/models/legacy/dco.pth", "#7f7f7f"),
    ("Peak term from the first epoch", "erp_forward/models/100k/dco.pth", "#d62728"),
    ("Staged peak term (from 80% of the epochs)", "erp_forward/models/100k/dco_sorted_phys.pth", "#2ca02c"),
]

ps = dict(load_operator_checkpoint(MODELS[1][1])["preprocessing_state"])
ps["dataset_file"] = FILES
dataset, _ = prepare_operator_data(num_configurations=len(ps["selected_source_ids"]), batch_size=128, dataset_file=FILES,
                                   preprocessing_state=ps, verbose=False)
test_id = int(_split_ids(dataset)["test"][CONFIG - 1])
configuration = _configuration_features(dataset)[test_id].copy()
print("configuration [m, k, f_t, x, y]:\n", np.round(configuration, 3))

results = []
for title, path, colour in MODELS:
    ck = load_operator_checkpoint(path)
    model = build_operator_model(dco.build_model, dataset.num_res, ck["model_config"],
                                 norm_params=ck["preprocessing_state"]["norm_params"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    out = predict_erp_spectrum(model, configuration, ck["preprocessing_state"]["norm_params"],
                               frequency_values=np.asarray(dataset.frequency_values), compare_solver=True,
                               plot=False, save_plots=False)
    err = out["prediction"] - out["ground_truth"]
    results.append((title, colour, out, float((err ** 2).mean()), float(np.abs(err).mean())))
    print(f"{title}: MSE {results[-1][3]:.3f} dB^2, MAE {results[-1][4]:.3f} dB")

freq = results[0][2]["frequencies"]
truth = results[0][2]["ground_truth"]
fig = plt.figure(figsize=(21, 4.8))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.75], wspace=0.22)
for i, (title, colour, out, mse, mae) in enumerate(results):
    ax = fig.add_subplot(gs[0, i])
    ax.plot(freq, truth, color="black", lw=2.6, label="Ground truth (solver)")
    ax.plot(freq, out["prediction"], color=colour, lw=1.8, ls="--", label="Prediction")
    for ft in configuration[:, 2]:
        ax.axvline(ft, color="#c2412c", ls=":", lw=1.0)
    ax.set_title(f"{title}\nMSE $={mse:.1f}$ dB$^2$, MAE $={mae:.2f}$ dB", fontsize=12)
    ax.set_xlabel("Frequency (Hz)")
    ax.grid(True, color="#d9d9d4", lw=0.5)
    ax.set_ylim(58, 133)
    if i == 0:
        ax.set_ylabel("ERP (dB)")
        ax.legend(loc="lower right", frameon=False, fontsize=10)
ax = fig.add_subplot(gs[0, 3])
ax.add_patch(plt.Rectangle((0, 0), Lx, Ly, fill=False, ec="black", lw=1.6))
ax.plot([xf], [yf], marker="*", color="#35d0ff", ms=18, mec="black", ls="", label="Force")
ax.plot(configuration[:, 3], configuration[:, 4], "o", color="#dc143c", ms=11, mec="black", ls="", label="Resonators")
for m, k, ft, x, y in configuration:
    ax.annotate(f"{ft:.0f} Hz", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=10)
ax.set_xlim(-0.05, Lx + 0.05); ax.set_ylim(-0.05, Ly + 0.05); ax.set_aspect("equal")
ax.set_xlabel("Position $x$ (m)"); ax.set_ylabel("Position $y$ (m)"); ax.set_title(f"Test configuration {CONFIG}", fontsize=12)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False, fontsize=10)
for ext, kw in (("png", dict(dpi=200, bbox_inches="tight")), ("pdf", dict(bbox_inches="tight"))):
    fig.savefig(ROOT / "presentation_figures" / f"peak_schedule_config{CONFIG}.{ext}", facecolor="white", **kw)
print("saved")
