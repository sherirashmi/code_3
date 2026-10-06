"""True vs predicted ERP of the 10 forward operators (final 200-epoch checkpoints in erp_forward/models/100k) on cases that were NOT used for
training: one test-split configuration, two new in-range hand-picked configurations, one outside the training ranges.
Ground truth = plate solver (150 modes, as in the 100k dataset). Inference only; saves forward_case_<n>.png, forward_cases.csv and
forward_novelty.txt into presentation_figures/prediction_cases/.

    python presentation_figures/prediction_cases/forward_cases.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403  (also puts the repo root on sys.path)

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.plot_style  # noqa: F401
from erp_forward.scripts.neural_operator_utils import _configuration_features, _split_ids, prepare_operator_data
from erp_forward.scripts.predict_custom import load_operator, predict
from utils.erp_dataset import DATASETS, configuration_to_resonators, denormalize_erp_array, select_dataset_modal_resolution
from utils.paths import forward_model_path
from utils.physics import freqs
from utils.solver import compute_erp_spectrum

DATASET = "100k"
TEST_INDEX = 3  # 1-based index into the test split
# short name, final checkpoint (erp_forward/models/100k/<name>.pth), colour, dashed
MODELS = [
    ("DON", "don_sorted_phys", "#1baf7a", False), ("DNO", "dno_sorted_phys", "#eb6834", False),
    ("DCO", "dco_sorted_phys", "#2a78d6", False), ("FNO", "fno_sorted_phys", "#eda100", False),
    ("WNO", "wno_sorted_phys", "#1baf7a", True), ("LNO", "lno_sorted_phys", "#008300", False),
    ("GNO", "gno_phys", "#e87ba4", False), ("STO", "sto_phys", "#4a3aa7", True),
    ("SIREN", "siren_sorted_phys", "#e34948", False), ("NN", "nn_perm", "#4a3aa7", False),
]
NEW_CASES = [  # (title, [(m [kg], f_t [Hz], x [m], y [m])]); none of them is in the dataset
    ("New configuration inside the training ranges (hand-picked, not in the dataset)",
     [(0.40, 45.0, 0.30, 0.12), (0.70, 95.0, 1.10, 0.38), (0.25, 130.0, 0.65, 0.25)]),
    ("All three resonators tuned to the same frequency (72 Hz), placed along one line",
     [(0.30, 72.0, 0.30, 0.20), (0.50, 72.0, 0.80, 0.20), (0.70, 72.0, 1.20, 0.20)]),
    ("Outside the training ranges: heavy resonators, m = 2.0, 1.5 and 2.5 kg (training: 0.1-1 kg)",
     [(2.00, 40.0, 0.45, 0.25), (1.50, 85.0, 1.10, 0.20), (2.50, 125.0, 0.70, 0.40)]),
]

select_dataset_modal_resolution(DATASET)
files = list(DATASETS[DATASET]["files"])
dataset, loaders = prepare_operator_data(num_configurations=int(DATASETS[DATASET]["num_configurations"]), batch_size=64, dataset_file=files, seed=727, verbose=False)
norm0 = dataset.norm_params
ids = _split_ids(dataset)["test"]
test_cfg = _configuration_features(dataset)[ids[TEST_INDEX - 1]].astype(np.float32)  # (R, 5) m, k, f_t, x, y
test_truth = np.asarray(dataset.responses, dtype=np.float64)[ids[TEST_INDEX - 1], :, 0]  # solver ERP in dB as stored in the dataset

cases = [(f"Test-split configuration {TEST_INDEX} (held out: not used for training)", test_cfg, test_truth)]
frequency_values = np.asarray(freqs, dtype=np.float64)
for title, rows in NEW_CASES:
    d = build_configuration(rows)
    cases.append((title, d, compute_erp_spectrum(configuration_to_resonators(d), frequencies=frequency_values)))

designs = np.stack([c[1] for c in cases])
dist, ref = training_distance(dataset, designs)
loaded = {label: load_operator(name) for label, name, _, _ in MODELS}
rows_csv, lines = [], ["Checkpoints used (erp_forward/models/100k/):"]
lines += [f"  {label:<6} {forward_model_path(name, DATASET)}" for label, name, _, _ in MODELS]
lines += ["", f"Novelty: distance of each case to its nearest TRAINING configuration (RMS over the range-scaled numbers m, f_t, x, y of the sorted resonators;"
          f" 0 = identical to a training sample). Median of the same distance for held-out test configurations: {ref:.3f}", ""]

for n, (title, d, truth) in enumerate(cases, 1):
    preds = {label: predict(model, norm, d[None], frequency_values.astype(np.float32))[0] for label, (model, norm) in loaded.items()}
    notes = out_of_range(d)
    lines.append(f"Case {n}: {title}" + (f"  [outside: {'; '.join(notes)}]" if notes else ""))
    lines += [f"  R{r + 1}: m={m:.2f} kg  k={k:,.0f} N/m  f_t={f_t:.0f} Hz  (x, y)=({x:.2f}, {y:.2f}) m" for r, (m, k, f_t, x, y) in enumerate(d)]
    lines.append(f"  nearest training configuration: {dist[n - 1]:.3f}  ({dist[n - 1] / ref:.1f} x the typical test configuration)")
    for label in preds:
        r_, p_ = rmse(preds[label], truth), peak_rmse(preds[label], truth)
        rows_csv.append([n, title, label, round(r_, 3), round(p_, 3)])
        lines.append(f"  {label:<6} RMSE {r_:6.2f} dB   peak RMSE {p_:6.2f} dB")
    lines.append("")

    fig = plt.figure(figsize=(19, 9.6))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.15, 1, 1], hspace=0.5, wspace=0.26, left=0.05, right=0.99, top=0.9, bottom=0.07)
    axp = fig.add_subplot(gs[0, :2])
    draw_plate(axp, {"True": d})
    axp.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
    axb = fig.add_subplot(gs[0, 3:])
    order = sorted(preds, key=lambda k: rmse(preds[k], truth))
    colour = {label: c for label, _, c, _ in MODELS}
    axb.barh(order[::-1], [rmse(preds[k], truth) for k in order][::-1], color=[colour[k] for k in order][::-1])
    for y_, k in enumerate(order[::-1]):
        axb.text(rmse(preds[k], truth), y_, f" {rmse(preds[k], truth):.2f}", va="center", fontsize=9)
    axb.set_xlabel("RMSE vs solver (dB)")
    axb.grid(axis="x", alpha=0.3)
    for i, (label, _, c, dashed) in enumerate(MODELS):
        ax = fig.add_subplot(gs[1 + i // 5, i % 5])
        ax.plot(frequency_values, truth, color="black", lw=2.0, label="True (solver)")
        ax.plot(frequency_values, preds[label], color=c, lw=1.6, ls="--" if dashed else "-", label=f"{label} predicted")
        for ft in d[:, 2]:
            if fmin <= ft <= fmax:
                ax.axvline(ft, color="#c2412c", ls="-.", lw=0.7, zorder=0)
        ax.set_title(f"{label}  (RMSE {rmse(preds[label], truth):.2f} dB)", fontsize=11)
        ax.set_xlim(fmin, fmax)
        ax.grid(alpha=0.3)
        if i % 5 == 0:
            ax.set_ylabel("ERP (dB)")
        if i >= 5:
            ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Case {n}: {title}" + (f"\n(outside the training set: {'; '.join(notes)})" if notes else ""), fontsize=13)
    fig.savefig(OUT / f"forward_case_{n}.png", dpi=170, facecolor="white")
    plt.close(fig)

(OUT / "forward_novelty.txt").write_text("\n".join(lines) + "\n")
with open(OUT / "forward_cases.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["case", "title", "model", "rmse_db", "peak_rmse_db"])
    w.writerows(rows_csv)
print("\n".join(lines))
