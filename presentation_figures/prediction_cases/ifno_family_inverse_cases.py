"""Inverse collage of the iFNO family (iFNO, iDCO, iGNO, iDNO, iLNO, iSTO; erp_invertible/models/200k_2res_18modes) on cases that were NOT used for
training: the plate with the true resonators and the design each model recovered, the RMSE of the recovered designs (solver ERP vs target) and
the target ERP against the ERP of the recovered designs, one panel per model. Inference only; ground truth = plate solver (18 modes, 2 resonators).

For every target each model draws 16 designs (seed 727). "Selected" = the draw whose OWN forward prediction is closest to the target (model's
forward direction, solver-free, as in erp_invertible/scripts/train.py); "best of 16" = the draw whose solver ERP is closest (an oracle choice that
needs the solver and the target). Rendered with real LaTeX when it is installed (utils/plot_style.py).
Saves png and pdf (vector) of ifno_family_inverse_case_<n>.png and ifno_family_inverse.csv in this folder.

    python presentation_figures/prediction_cases/ifno_family_inverse_cases.py
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

import utils.plot_style  # noqa: F401
import common as _common
from utils.plot_style import save_figure

_common.RES_COLORS = ("#111111", "#9a9a9a")  # true resonators: black / grey (the model colours are blue, orange, ...)
from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_invertible.scripts.train import RECOMMENDED_OPTIONS, _load_checkpoint
from utils.erp_dataset import (DATASETS, configuration_to_resonators, denormalize_erp_array, normalize_configuration_array,
                               normalize_erp_array, select_dataset_modal_resolution)
from utils.physics import fmax, fmin
from utils.solver import compute_erp_spectrum
from utils.support import device

TAG = "200k_2res_18modes"
TEST_INDEX, S = 3, 16
MODELS = [("1", "iFNO", "#d62728"), ("2", "iDCO", "#8c564b"), ("3", "iGNO", "#ff7f0e"), ("4", "iDNO", "#9467bd"),
          ("6", "iLNO", "#2ca02c"), ("8", "iSTO", "#1f77b4")]
NEW_CASES = [
    ("New configuration inside the training ranges (hand-picked, not in the dataset)", [(0.45, 55.0, 0.35, 0.15), (0.65, 110.0, 1.05, 0.38)]),
    ("Outside the training ranges: heavy resonators, m = 2.0 and 1.5 kg (training: 0.1--1 kg)", [(2.00, 60.0, 0.45, 0.25), (1.50, 120.0, 1.05, 0.15)]),
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

models = {}
for key, label, _ in MODELS:
    m, _ck = _load_checkpoint(key, TAG, False, RECOMMENDED_OPTIONS)
    m.set_design_normalization(norm)
    models[label] = m.eval()
colour = {label: c for _, label, c in MODELS}


def solve(c):
    return compute_erp_spectrum(configuration_to_resonators(np.asarray(c, dtype=np.float64)), frequencies=freq)


rows_csv = []
for n, (title, d, truth) in enumerate(cases, 1):
    spec = torch.from_numpy(normalize_erp_array(truth[None].astype(np.float32), norm)).to(device)
    res = {}
    for label, model in models.items():
        torch.manual_seed(727)
        with torch.no_grad():
            flat = model.sample(spec, num_samples=S)[0].cpu().numpy()  # (S, D)
            physical = denormalize_design(flat, num_res, norm)  # (S, R, 5)
            renorm = torch.from_numpy(normalize_configuration_array(physical, norm)).to(device)
            own = model.predict_spectrum(renorm).squeeze(-1).cpu()  # (S, F) normalised
        sel = int(((own - spec.cpu()) ** 2).mean(-1).argmin())
        solved = np.stack([solve(c) for c in physical])
        err = np.sqrt(((solved - truth) ** 2).mean(1))
        best = int(err.argmin())
        phys_sorted = sort_resonators_by_ft(physical)
        res[label] = dict(solved=solved, err=err, sel=sel, best=best, design=phys_sorted[sel],
                          pos=100 * np.hypot(phys_sorted[sel][:, 3] - d[:, 3], phys_sorted[sel][:, 4] - d[:, 4]))
        rows_csv.append([n, title, label, round(float(err[sel]), 3), round(float(err[best]), 3), [round(float(v), 1) for v in res[label]["pos"]]])
    print(f"Case {n}: {title}")
    for label, r in res.items():
        print(f"  {label:<5} selected {r['err'][r['sel']]:6.2f} dB | best of {S} {r['err'][r['best']]:6.2f} dB | position error {np.round(r['pos'], 0)} cm")

    fig = plt.figure(figsize=(18, 12.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1], hspace=0.66, wspace=0.2, left=0.05, right=0.99, top=0.9, bottom=0.07)
    axp = fig.add_subplot(gs[0, 0])
    draw_plate(axp, {"True": d})
    for label, r in res.items():
        for i in range(num_res):
            axp.plot([d[i, 3], r["design"][i, 3]], [d[i, 4], r["design"][i, 4]], color=colour[label], lw=0.8, ls=":", zorder=3)
        axp.plot(r["design"][:, 3], r["design"][:, 4], "X", color=colour[label], ms=10, mec="black", mew=0.6, ls="", zorder=6)
    h, lab = axp.get_legend_handles_labels()
    h += [Line2D([], [], marker="X", color=colour[l], mec="black", mew=0.6, ls="", ms=9) for l in res]
    lab += [f"{l} (selected design)" for l in res]
    axp.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False, fontsize=8.6, ncol=2, columnspacing=1.0)
    axb = fig.add_subplot(gs[0, 1:])
    order = sorted(res, key=lambda k: res[k]["err"][res[k]["sel"]])[::-1]
    ypos = np.arange(len(order))
    axb.barh(ypos + 0.19, [res[k]["err"][res[k]["sel"]] for k in order], height=0.36, color=[colour[k] for k in order], label="selected (own forward)")
    axb.barh(ypos - 0.19, [res[k]["err"][res[k]["best"]] for k in order], height=0.36, color=[colour[k] for k in order], alpha=0.45, hatch="//",
             edgecolor="white", label=f"best of {S} (solver, oracle)")
    for y_, k in zip(ypos, order):
        axb.text(res[k]["err"][res[k]["sel"]], y_ + 0.19, f" {res[k]['err'][res[k]['sel']]:.2f}", va="center", fontsize=9)
        axb.text(res[k]["err"][res[k]["best"]], y_ - 0.19, f" {res[k]['err'][res[k]['best']]:.2f}", va="center", fontsize=9)
    axb.set_yticks(ypos, order)
    axb.set_xlabel("RMSE of the ERP of the recovered design vs the target (dB)")
    axb.grid(axis="x", alpha=0.3)
    axb.set_xlim(0, max(max(r["err"][r["sel"]], r["err"][r["best"]]) for r in res.values()) * 1.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#777777"), plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.45, hatch="//")]
    axb.legend(handles, ["selected (own forward)", f"best of {S} (solver, oracle)"], loc="upper right", fontsize=9)
    for i, (label, r) in enumerate(res.items()):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        for s in range(S):
            if s not in (r["sel"], r["best"]):
                ax.plot(freq, r["solved"][s], color="#b9bec6", lw=0.6, zorder=1)
        ax.plot(freq, truth, color="black", lw=2.0, zorder=4, label="Target ERP")
        ax.plot(freq, r["solved"][r["sel"]], color=colour[label], lw=1.7, zorder=3, label=f"selected ({r['err'][r['sel']]:.2f} dB)")
        if r["best"] != r["sel"]:
            ax.plot(freq, r["solved"][r["best"]], color=colour[label], lw=1.6, ls="--", zorder=2, label=f"best of {S} ({r['err'][r['best']]:.2f} dB)")
        ax.set_title(label, fontsize=12)
        ax.set_xlim(fmin, fmax)
        ax.grid(alpha=0.3)
        if i % 3 == 0:
            ax.set_ylabel("ERP (dB)")
        if i >= 3:
            ax.set_xlabel("Frequency (Hz)")
        ax.legend(fontsize=7.6, loc="lower right")
    notes = out_of_range(d)
    fig.suptitle(f"iFNO family, inverse direction, case {n}: {title}" + (f"\n(outside the training set: {'; '.join(notes)})" if notes else ""), fontsize=12.5)
    save_figure(fig, OUT / f"ifno_family_inverse_case_{n}.png", dpi=170, close=False)
    save_figure(fig, OUT / f"ifno_family_inverse_case_{n}.pdf")  # vector, fonts embedded

with open(OUT / "ifno_family_inverse.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["case", "title", "model", "selected_rmse_db", "best_of_16_rmse_db", "selected_position_error_cm"])
    w.writerows(rows_csv)
