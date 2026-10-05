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
from utils.support import device

CONFIG = 4  # 1-based test configuration
FILES = ["datasets/erp/3res/100k/dataset_erp_ft_100k_part1.pth", "datasets/erp/3res/100k/dataset_erp_ft_100k_part2.pth"]
MODELS = [  # title, checkpoint, colour
    ("Loss equation (MSE + slope)", "erp_forward/models/legacy/dco.pth", "#7f7f7f"),
    ("Added peak term in the loss", "erp_forward/models/100k/dco.pth", "#d62728"),
    ("Peak term included from 80% of the epochs", "erp_forward/models/100k/dco_sorted_phys.pth", "#2ca02c"),
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
NAMES = ["1_mse_slope", "2_peak_term_added", "3_peak_term_from_80_percent"]
for name, (title, colour, out, mse, mae) in zip(NAMES, results):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(freq, truth, color="black", lw=2.6, label="Ground truth (solver)")
    ax.plot(freq, out["prediction"], color=colour, lw=1.9, ls="--", label="Prediction")
    for ft in configuration[:, 2]:
        ax.axvline(ft, color="#c2412c", ls=":", lw=1.0)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ERP (dB)")
    ax.grid(True, color="#d9d9d4", lw=0.5)
    ax.set_ylim(58, 133)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout()
    for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
        fig.savefig(ROOT / "presentation_figures" / f"peak_schedule_config{CONFIG}_{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
print("saved")
