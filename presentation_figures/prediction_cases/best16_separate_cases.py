"""Separate best-of-16 graph for every iFNO-family model and every case of the inverse collages (ifno_family_inverse_case_<n>.png), in the style of
presentation_figures/invertible_best16_<model>.png: target ERP (black), the best of 16 sampled designs checked with the plate solver (dashed, the
model's colour) and the true tuning frequencies; plus one plate per case with the true resonators and the best design of every model.
"Best of 16" is an oracle choice: the draw whose solver ERP is closest to the target (needs the solver and the target). Same draws (seed 727) and
cases as ifno_family_inverse_cases.py, rendered with real LaTeX when installed. Inference only.
Saves best16_separate/case_<n>/<model>.png|pdf and best16_separate/case_<n>/plate.png|pdf in this folder.

    python presentation_figures/prediction_cases/best16_separate_cases.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import utils.plot_style  # noqa: F401
from utils.plot_style import save_figure
from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_invertible.scripts.train import RECOMMENDED_OPTIONS, _load_checkpoint
from utils.erp_dataset import DATASETS, configuration_to_resonators, denormalize_erp_array, normalize_erp_array, select_dataset_modal_resolution
from utils.physics import Lx, Ly, fmax, fmin, xf, yf
from utils.solver import compute_erp_spectrum
from utils.support import device

TAG = "200k_2res_18modes"
TEST_INDEX, S = 3, 16
MODELS = [("1", "iFNO", "#d62728"), ("2", "iDCO", "#8c564b"), ("3", "iGNO", "#ff7f0e"), ("4", "iDNO", "#9467bd"),
          ("6", "iLNO", "#2ca02c"), ("8", "iSTO", "#1f77b4")]
NEW_CASES = [
    ("New configuration inside the training ranges (hand-picked, not in the dataset)", [(0.45, 55.0, 0.35, 0.15), (0.65, 110.0, 1.05, 0.38)]),
    ("Outside the training ranges: heavy resonators, m = 2.0 and 1.5 kg", [(2.00, 60.0, 0.45, 0.25), (1.50, 120.0, 1.05, 0.15)]),
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
cases = [(f"Test-split configuration {TEST_INDEX}", test_design, denormalize_erp_array(spectrum[i0 : i0 + 1], norm)[0])]
for title, rows in NEW_CASES:
    d = sort_resonators_by_ft(build_configuration(rows))
    cases.append((title, d, compute_erp_spectrum(configuration_to_resonators(d), frequencies=freq)))

models = {}
for key, label, _ in MODELS:
    m, _ck = _load_checkpoint(key, TAG, False, RECOMMENDED_OPTIONS)
    m.set_design_normalization(norm)
    models[label] = m.eval()
colour = {label: c for _, label, c in MODELS}


def solve(c):
    return compute_erp_spectrum(configuration_to_resonators(np.asarray(c, dtype=np.float64)), frequencies=freq)


def save_both(fig, base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, base.with_suffix(".png"), dpi=200, close=False, verbose=False)
    save_figure(fig, base.with_suffix(".pdf"), verbose=False)


rows_csv = []
for n, (title, d, truth) in enumerate(cases, 1):
    spec = torch.from_numpy(normalize_erp_array(truth[None].astype(np.float32), norm)).to(device)
    best = {}
    for label, model in models.items():
        torch.manual_seed(727)
        with torch.no_grad():
            flat = model.sample(spec, num_samples=S)[0].cpu().numpy()
        physical = sort_resonators_by_ft(denormalize_design(flat, num_res, norm))
        solved = np.stack([solve(c) for c in physical])
        err = np.sqrt(((solved - truth) ** 2).mean(1))
        b = int(err.argmin())
        best[label] = dict(curve=solved[b], err=float(err[b]), design=physical[b])
        rows_csv.append([n, title, label, round(float(err[b]), 3)])
    lo = min(float(truth.min()), *(float(r["curve"].min()) for r in best.values())) - 3
    hi = max(float(truth.max()), *(float(r["curve"].max()) for r in best.values())) + 3
    out = OUT / "best16_separate" / f"case_{n}"
    for label, r in best.items():
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        ax.plot(freq, truth, color="black", lw=2.8, label="Target ERP")
        ax.plot(freq, r["curve"], color=colour[label], lw=1.9, ls="--", label=f"{label}, best of {S} (RMSE {r['err']:.2f} dB)")
        for j, ft in enumerate(d[:, 2]):
            ax.axvline(ft, color="#c2412c", ls="--", lw=1.2, zorder=1, label="True tuning frequencies" if j == 0 else None)
        ax.set_xlim(fmin, fmax)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("ERP (dB)")
        ax.set_title(label, fontsize=14)
        ax.grid(True, color="#d9d9d4", lw=0.5)
        ax.legend(frameon=False, loc="lower right", fontsize=10)
        fig.tight_layout()
        save_both(fig, out / label)
    fig, axp = plt.subplots(figsize=(9.5, 4.8))
    axp.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, ec="black", lw=1.6))
    axp.plot([xf], [yf], marker="*", color="#35d0ff", ms=18, mec="black", ls="", zorder=5)
    for label, r in best.items():
        for i in range(num_res):
            axp.plot([d[i, 3], r["design"][i, 3]], [d[i, 4], r["design"][i, 4]], color=colour[label], lw=0.9, ls=":", zorder=3)
        axp.plot(r["design"][:, 3], r["design"][:, 4], "X", color=colour[label], ms=11, mec="black", mew=0.6, ls="", zorder=6)
    axp.plot(d[:, 3], d[:, 4], "o", color="#dc143c", ms=13, mec="black", ls="", zorder=7)
    for i in range(num_res):
        axp.annotate(f"R{i + 1}", (d[i, 3], d[i, 4]), xytext=(-4, 9), textcoords="offset points", fontsize=12, ha="right", zorder=8)
    axp.set_xlim(-0.05, Lx + 0.05)
    axp.set_ylim(-0.05, Ly + 0.05)
    axp.set_aspect("equal")
    axp.set_xlabel("Position $x$ (m)")
    axp.set_ylabel("Position $y$ (m)")
    handles = [Line2D([], [], marker="*", color="#35d0ff", mec="black", ls="", ms=14, label="Force"),
               Line2D([], [], marker="o", color="#dc143c", mec="black", ls="", ms=10, label="True resonator")]
    handles += [Line2D([], [], marker="X", color=colour[l], mec="black", mew=0.6, ls="", ms=10, label=l) for l in best]
    axp.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=10, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout()
    save_both(fig, out / "plate")
    print(f"case {n} ({title}): " + ", ".join(f"{l} {r['err']:.2f}" for l, r in best.items()), flush=True)

with open(OUT / "best16_separate" / "best16_rmse.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["case", "title", "model", "best_of_16_rmse_db"])
    w.writerows(rows_csv)
