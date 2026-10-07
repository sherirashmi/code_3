"""Forward collage of the iFNO family (iFNO, iDCO, iGNO, iDNO, iLNO, iSTO: the six trained in erp_invertible/models/200k_2res_18modes): for
cases that were NOT used for training the plate with the configuration, the RMSE of every model and the true (solver) vs predicted ERP of
each model (design -> ERP). Inference only; ground truth = plate solver (18 modes, 2 resonators). The outside-the-ranges case of
inverse_cases.py has no forward panel because the models encode the design in bounded coordinates (m <= 1 kg). Rendered with real LaTeX when installed (utils/plot_style.py).
Saves png and pdf (vector) of ifno_family_forward_case_<n>.png and ifno_family_forward.csv in this folder.

    python presentation_figures/prediction_cases/ifno_family_forward_cases.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.plot_style  # noqa: F401
from utils.plot_style import save_figure
from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_inverse.scripts.design_space import encode_bounded
from erp_invertible.scripts.train import RECOMMENDED_OPTIONS, _load_checkpoint, variant
from utils.erp_dataset import DATASETS, configuration_to_resonators, denormalize_erp_array, select_dataset_modal_resolution
from utils.paths import invertible_model_path
from utils.physics import fmax, fmin
from utils.solver import compute_erp_spectrum
from utils.support import device

TAG = "200k_2res_18modes"
TEST_INDEX = 3
# key in the registry, label, colour
MODELS = [("1", "iFNO", "#d62728"), ("2", "iDCO", "#8c564b"), ("3", "iGNO", "#ff7f0e"), ("4", "iDNO", "#9467bd"),
          ("6", "iLNO", "#2ca02c"), ("8", "iSTO", "#1f77b4")]
NEW_CASES = [
    ("New configuration inside the training ranges (hand-picked, not in the dataset)", [(0.45, 55.0, 0.35, 0.15), (0.65, 110.0, 1.05, 0.38)]),
    ("Different ERP: heavy absorbers on the first two plate peaks (46 and 55 Hz)", [(1.00, 46.0, 0.45, 0.25), (0.90, 55.0, 0.90, 0.28)]),
    ("Different ERP: heavy absorbers on the two high plate peaks (98 and 145 Hz)", [(1.00, 98.0, 0.60, 0.20), (1.00, 145.0, 1.10, 0.30)]),
    ("Different ERP: both resonators tuned to 100 Hz, heavy", [(1.00, 100.0, 0.55, 0.20), (1.00, 100.0, 0.90, 0.30)]),
]

select_dataset_modal_resolution(TAG)
files = list(DATASETS[TAG]["files"])
dataset, loaders = prepare_inverse_data(num_configurations=int(DATASETS[TAG]["num_configurations"]), batch_size=128,
                                        dataset_file=files if len(files) > 1 else files[0], seed=727, design_param="bounded12")
norm = dataset.norm_params
num_res = int(norm["num_res"])
freq = np.asarray(dataset.frequency_values, dtype=np.float64)
spectrum, design = next(iter(loaders["test"]))
i0 = TEST_INDEX - 1
test_design = sort_resonators_by_ft(denormalize_design(design[i0 : i0 + 1].reshape(1, -1).numpy(), num_res, norm, consistent=False))[0]
cases = [(f"Test-split configuration {TEST_INDEX} (held out: not used for training)", test_design, denormalize_erp_array(spectrum[i0 : i0 + 1], norm)[0])]
for title, rows in NEW_CASES:
    d = sort_resonators_by_ft(build_configuration(rows))
    cases.append((title, d, compute_erp_spectrum(configuration_to_resonators(d), frequencies=freq)))
dist, ref = training_distance(dataset, np.stack([c[1] for c in cases]))

models, paths = {}, {}
for key, label, _ in MODELS:
    model, _ck = _load_checkpoint(key, TAG, False, RECOMMENDED_OPTIONS)
    model.set_design_normalization(norm)
    models[label] = model.eval()
    paths[label] = invertible_model_path(variant(key, False, RECOMMENDED_OPTIONS)[0], TAG)
colour = {label: c for _, label, c in MODELS}


@torch.no_grad()
def forward_erp(label, d):
    a = torch.from_numpy(encode_bounded(d, norm).reshape(1, -1)).to(device)
    return denormalize_erp_array(models[label].predict_spectrum_from_design(a).squeeze(-1).cpu().numpy(), norm)[0]


rows_csv = []
print("Checkpoints:", *[f"\n  {l}: {p}" for l, p in paths.items()])
print(f"novelty reference (median test-config distance to training set): {ref:.3f}")
for n, (title, d, truth) in enumerate(cases, 1):
    preds = {label: forward_erp(label, d) for label in models}
    print(f"\nCase {n}: {title} | nearest training configuration {dist[n - 1]:.3f} ({dist[n - 1] / ref:.1f} x typical)")
    for label in preds:
        r_, p_ = rmse(preds[label], truth), peak_rmse(preds[label], truth)
        rows_csv.append([n, title, label, round(r_, 3), round(p_, 3)])
        print(f"  {label:<5} RMSE {r_:6.2f} dB   peak RMSE {p_:6.2f} dB")
    fig = plt.figure(figsize=(18, 11.6))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=0.62, wspace=0.2, left=0.05, right=0.99, top=0.9, bottom=0.06)
    axp = fig.add_subplot(gs[0, 0])
    draw_plate(axp, {"True": d})
    axp.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False, fontsize=9.5, ncol=1)
    axb = fig.add_subplot(gs[0, 1:])
    order = sorted(preds, key=lambda k: rmse(preds[k], truth))
    axb.barh(order[::-1], [rmse(preds[k], truth) for k in order][::-1], color=[colour[k] for k in order][::-1])
    for y_, k in enumerate(order[::-1]):
        axb.text(rmse(preds[k], truth), y_, f" {rmse(preds[k], truth):.2f}", va="center", fontsize=10)
    axb.set_xlabel("RMSE vs solver (dB)")
    axb.grid(axis="x", alpha=0.3)
    axb.set_xlim(0, max(rmse(p, truth) for p in preds.values()) * 1.15)
    for i, label in enumerate(models):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        ax.plot(freq, truth, color="black", lw=2.0, label="True (solver)")
        ax.plot(freq, preds[label], color=colour[label], lw=1.7, label=f"{label} predicted")
        for ft in d[:, 2]:
            ax.axvline(ft, color="#c2412c", ls="-.", lw=0.7, zorder=0)
        ax.set_title(f"{label}  (RMSE {rmse(preds[label], truth):.2f} dB)", fontsize=12)
        ax.set_xlim(fmin, fmax)
        ax.grid(alpha=0.3)
        if i % 3 == 0:
            ax.set_ylabel("ERP (dB)")
        if i >= 3:
            ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(f"iFNO family, forward direction, case {n}: {title}", fontsize=13)
    save_figure(fig, OUT / f"ifno_family_forward_case_{n}.png", dpi=170, close=False)
    save_figure(fig, OUT / f"ifno_family_forward_case_{n}.pdf")  # vector, fonts embedded

with open(OUT / "ifno_family_forward.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["case", "title", "model", "rmse_db", "peak_rmse_db"])
    w.writerows(rows_csv)
