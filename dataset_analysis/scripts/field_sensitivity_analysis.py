"""One-at-a-time (OAT) global sensitivity analysis: which of the physical
design fields moves the true ERP spectrum the most, for a MATCHED
relative perturbation size?

Important caveat this script makes explicit: f_t is NOT an independent
physical input -- the solver (utils.solver.compute_coupled_matrices) only
consumes m, k, x, y directly; f_t = sqrt(k/m)/(2*pi) is a derived label.
So there is no separate "5th" perturbation channel for f_t in isolation --
shifting f_t necessarily means moving m (holding k fixed) or moving k
(holding m fixed), which are exactly the "m" and "k" tests below.

Method: sample many real baseline configurations from the 100k dataset,
perturb ONE field of ONE resonator at a time by +-5% of that field's own
dataset-generation range (clipped to stay in-range), re-solve the TRUE
ERP spectrum, and record the resulting change. Averaging the two signed
directions and many random baselines gives a robust, apples-to-apples
comparison across m, k, x, y (all nudged by the same FRACTION of their
own physical range, so a variable with a huge absolute range like k isn't
unfairly favored/penalized).
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/user/code_3")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.erp_dataset import ERPDataset
from utils.physics import Lx, Ly, edge_margin, fmax, fmin, freqs, m_max, m_min
from utils.solver import compute_erp_spectrum

OUT_PATH = "dataset_analysis/plots/field_sensitivity.png"

rng = np.random.default_rng(0)

M_LO, M_HI = m_min, m_max
X_LO, X_HI = edge_margin, Lx - edge_margin
Y_LO, Y_HI = edge_margin, Ly - edge_margin
K_LO, K_HI = m_min * (2 * np.pi * fmin) ** 2, m_max * (2 * np.pi * fmax) ** 2

FIELDS = {"m": (M_LO, M_HI), "k": (K_LO, K_HI), "x": (X_LO, X_HI), "y": (Y_LO, Y_HI)}
REL_STEP = 0.05  # 5% of each field's own dataset-generation range

N_BASELINES = 25
RES_IDX = 1  # which of the 3 resonators gets perturbed

print("Loading dataset for realistic baseline configurations...")
dataset = ERPDataset().load_shards([
    "datasets/dataset_erp_ft_100k_part1.pth",
    "datasets/dataset_erp_ft_100k_part2.pth",
])
configs_all = np.asarray(dataset.configuration_features, dtype=np.float64)  # (N,3,5) [m,k,f_t,x,y]
N = configs_all.shape[0]
baseline_ids = rng.choice(N, size=N_BASELINES, replace=False)

results = {f: [] for f in FIELDS}
field_to_col = {"m": 0, "k": 1, "x": 3, "y": 4}

print(f"Running OAT sensitivity over {N_BASELINES} baselines x {len(FIELDS)} fields x 2 directions...")
t0 = time.time()
n_solves = 0
for bi, cid in enumerate(baseline_ids):
    cfg = configs_all[cid].copy()
    resonators_base = [{"m": cfg[r, 0], "k": cfg[r, 1], "x": cfg[r, 3], "y": cfg[r, 4]} for r in range(3)]
    erp_base = compute_erp_spectrum(resonators_base, frequencies=freqs)
    n_solves += 1

    for field, (lo, hi) in FIELDS.items():
        col = field_to_col[field]
        step = REL_STEP * (hi - lo)
        deltas = []
        for sign in (+1, -1):
            perturbed = cfg.copy()
            perturbed[RES_IDX, col] = np.clip(perturbed[RES_IDX, col] + sign * step, lo, hi)
            resonators = [{"m": perturbed[r, 0], "k": perturbed[r, 1], "x": perturbed[r, 3], "y": perturbed[r, 4]} for r in range(3)]
            erp_pert = compute_erp_spectrum(resonators, frequencies=freqs)
            n_solves += 1
            mse = ((erp_pert - erp_base) ** 2).mean()
            maxabs = np.abs(erp_pert - erp_base).max()
            deltas.append((mse, maxabs))
        results[field].append(deltas)

    if (bi + 1) % 5 == 0:
        print(f"  baseline {bi + 1}/{N_BASELINES}  ({time.time() - t0:.1f}s elapsed, {n_solves} solves)")

print(f"\nTotal solves: {n_solves}, total time: {time.time() - t0:.1f}s")
print(f"\nOAT sensitivity: resonator #{RES_IDX}'s field nudged by +-{REL_STEP * 100:.0f}% of its own range,")
print(f"other 3 resonators + other fields held fixed. Averaged over {N_BASELINES} real baselines x 2 directions.")
print()
print(f"{'field':>6} {'range':>22} {'step (+-)':>12} {'mean ERP MSE':>14} {'mean max|delta|':>16}")
summary = {}
for field, (lo, hi) in FIELDS.items():
    all_mse = [d[0] for pair in results[field] for d in pair]
    all_max = [d[1] for pair in results[field] for d in pair]
    summary[field] = (np.mean(all_mse), np.mean(all_max), np.std(all_mse, ddof=1) / np.sqrt(len(all_mse)))
    step = REL_STEP * (hi - lo)
    print(f"{field:>6} [{lo:8.2f},{hi:8.2f}] {step:12.3f} {summary[field][0]:14.4f} {summary[field][1]:16.4f}")

print()
ranked = sorted(summary.items(), key=lambda kv: kv[1][0], reverse=True)
print("Ranked by mean ERP MSE (most influential first):")
for rank, (field, (mse, maxabs, sem)) in enumerate(ranked, 1):
    print(f"  {rank}. {field}  (MSE={mse:.4f} dB^2, max|delta|={maxabs:.3f} dB)")

fields_ranked = [f for f, _ in ranked]
mses = [summary[f][0] for f in fields_ranked]
sems = [summary[f][2] for f in fields_ranked]
colors_map = {"k": "#C44E52", "x": "#DD8452", "m": "#4C72B0", "y": "#55A868"}
colors = [colors_map[f] for f in fields_ranked]

fig, ax = plt.subplots(figsize=(7.5, 5))
bars = ax.bar(fields_ranked, mses, yerr=sems, capsize=5, color=colors, edgecolor="black", linewidth=0.8)
ax.set_ylabel("Mean ERP spectrum MSE (dB²)\nfrom a ±5% (of range) one-at-a-time perturbation")
ax.set_title("Sensitivity of the true ERP spectrum to each design field\n"
              f"({N_BASELINES} real baselines x 2 directions, matched relative step size)")
ax.grid(axis="y", alpha=0.3)
for bar, mse_val in zip(bars, mses):
    ax.annotate(f"{mse_val:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
plt.close(fig)
print(f"\nSaved {OUT_PATH}")
