"""Search the full 100k ERP dataset for configurations whose TRUE spectra
are near-identical despite substantially different designs -- an
empirical, data-driven check of whether the inverse map (spectrum ->
design) is genuinely many-to-one, rather than reasoning about it
abstractly.

Pipeline:
  1. Load all 100k (spectrum, design) pairs.
  2. PCA-reduce the 301-dim ERP curves (they need K~211 components for
     99.5% variance -- these spectra are not especially low-dimensional)
     so nearest-neighbor search over 100k points is tractable.
  3. Chunked brute-force nearest-neighbor search in PCA space (never
     materializes a full 100k x 100k matrix).
  4. Re-score every candidate pair with the EXACT, full-resolution ERP
     curves (MSE in dB^2) -- PCA space is only used to screen candidates.
  5. Also compute a normalized design-space distance (canonicalized by
     f_t to remove the resonator-label permutation ambiguity) so
     "genuinely different design, same spectrum" can be separated from
     "nearly-identical design, same spectrum" (the latter is not
     interesting -- dense LHS sampling trivially puts nearby designs
     close in both spaces).
  6. Save two figures:
       - similar_erp_pairs.png: top individual pairs (design distance
         above the dataset's own median), overlaid ERP + both layouts.
       - similar_erp_clusters.png: whole GROUPS of >=3 mutually similar
         designs (found via connected components on the MSE<0.5 graph,
         then independently verified with the FULL pairwise MSE matrix
         within each group -- a connected component can be a "chain"
         where the endpoints aren't actually close, so this direct
         check is what confirms every member is close to every other
         member). Legends are drawn in a dedicated column per row so
         they never overlap the plotted curves/points.
"""
from __future__ import annotations

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.erp_dataset import ERPDataset
from utils.physics import Lx, Ly, xf, yf
from erp_inverse.scripts.common import canonicalize_by_ft

PAIRS_OUT = "dataset_analysis/100k/plots/similar_erp_pairs.png"
CLUSTERS_OUT = "dataset_analysis/100k/plots/similar_erp_clusters.png"

MSE_EDGE_THRESHOLD = 0.5  # dB^2, "near-identical" edge for building clusters
N_TOP_PAIRS = 5
MAX_CLUSTER_SIZE = 20
N_CLUSTERS_TO_PLOT = 6

# ---------------------------------------------------------
# Load + PCA + nearest-neighbor search
# ---------------------------------------------------------
print("Loading full 100k dataset...")
dataset = ERPDataset().load_shards([
    "datasets/erp/3res/100k/dataset_erp_ft_100k_part1.pth",
    "datasets/erp/3res/100k/dataset_erp_ft_100k_part2.pth",
])
N = dataset.num_samples
freqs = np.asarray(dataset.frequency_values, dtype=np.float64)
erp = np.asarray(dataset.responses, dtype=np.float64)[:, :, 0]  # (N, 301) dB
configs = canonicalize_by_ft(np.asarray(dataset.configuration_features, dtype=np.float64))
print(f"N={N}, erp shape={erp.shape}, configs shape={configs.shape}")

print("Computing PCA...")
mean = erp.mean(axis=0, keepdims=True)
centered = erp - mean
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
cum_explained = np.cumsum((S ** 2) / (S ** 2).sum())
K = max(int(np.searchsorted(cum_explained, 0.995)) + 1, 8)
print(f"Using K={K} components, explained variance={cum_explained[K - 1] * 100:.3f}%")
reduced = U[:, :K] * S[:K]

print("Nearest-neighbor search (chunked)...")
sq_norms = (reduced ** 2).sum(axis=1)
CHUNK = 2000
TOP_M = 5
best_idx = np.full((N, TOP_M), -1, dtype=np.int64)
for start in range(0, N, CHUNK):
    end = min(start + CHUNK, N)
    chunk = reduced[start:end]
    cross = chunk @ reduced.T
    d2 = sq_norms[start:end, None] + sq_norms[None, :] - 2.0 * cross
    np.maximum(d2, 0.0, out=d2)
    rows = np.arange(end - start)
    d2[rows, start + rows] = np.inf
    part = np.argpartition(d2, TOP_M, axis=1)[:, :TOP_M]
    part_d = np.take_along_axis(d2, part, axis=1)
    order = np.argsort(part_d, axis=1)
    best_idx[start:end] = np.take_along_axis(part, order, axis=1)
    if (start // CHUNK) % 10 == 0:
        print(f"  {end}/{N}")

print("Building + re-scoring candidate pairs...")
pairs = set()
for i in range(N):
    for j in best_idx[i]:
        if j >= 0:
            pairs.add((i, j) if i < j else (j, i))
pairs = np.array(sorted(pairs))
i_idx, j_idx = pairs[:, 0], pairs[:, 1]
erp_mse = ((erp[i_idx] - erp[j_idx]) ** 2).mean(axis=1)

flat_i = configs[i_idx].reshape(len(pairs), -1)
flat_j = configs[j_idx].reshape(len(pairs), -1)
field_std = np.tile(configs.reshape(-1, 5).std(axis=0), 3)
design_dist = np.sqrt((((flat_i - flat_j) / field_std) ** 2).sum(axis=1))
print(f"Candidate pairs: {len(pairs)}")

# ---------------------------------------------------------
# Figure 1: top individual pairs (design-distance filtered)
# ---------------------------------------------------------
MIN_DESIGN_DIST = np.percentile(design_dist, 40)
order = np.argsort(erp_mse)
top_pairs = []
for k in order:
    if design_dist[k] >= MIN_DESIGN_DIST:
        top_pairs.append((int(i_idx[k]), int(j_idx[k])))
    if len(top_pairs) >= N_TOP_PAIRS:
        break
print(f"\nTop {len(top_pairs)} 'different design, near-identical spectrum' pairs: {top_pairs}")

COLOR_A, COLOR_B = "#C44E52", "#4C72B0"
fig, axes = plt.subplots(len(top_pairs), 2, figsize=(13, 4.2 * len(top_pairs)),
                          gridspec_kw={"width_ratios": [1.6, 1]})
for row, (i, j) in enumerate(top_pairs):
    ax_erp, ax_plate = axes[row]
    erp_i, erp_j = erp[i], erp[j]
    mse = ((erp_i - erp_j) ** 2).mean()
    max_abs = np.abs(erp_i - erp_j).max()

    ax_erp.plot(freqs, erp_i, lw=2.2, color=COLOR_A, label=f"Design A (#{i})")
    ax_erp.plot(freqs, erp_j, lw=1.6, ls="--", color=COLOR_B, label=f"Design B (#{j})")
    ax_erp.set_xlabel("Frequency (Hz)")
    ax_erp.set_ylabel("ERP (dB)")
    ax_erp.set_title(f"Pair {row + 1}: MSE={mse:.4f} dB², max|Δ|={max_abs:.2f} dB", fontsize=11)
    ax_erp.grid(alpha=0.35)
    ax_erp.legend(loc="upper right", fontsize=8)

    ax_plate.add_patch(plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor="black", lw=1.5))
    ax_plate.scatter([xf], [yf], marker="*", s=160, color="cyan", edgecolors="black",
                      linewidths=0.8, zorder=4, label="Force F0")
    for label, idx, color, marker in (("A", i, COLOR_A, "o"), ("B", j, COLOR_B, "^")):
        cfg = configs[idx]
        ax_plate.scatter(cfg[:, 3], cfg[:, 4], marker=marker, s=100, color=color,
                          edgecolors="black", linewidths=0.8, zorder=3,
                          label=f"Design {label} resonators")
        for _m, _k, f_t, x, y in cfg:
            ax_plate.annotate(f"{f_t:.1f}", (x, y), textcoords="offset points",
                               xytext=(5, 5), fontsize=7, color=color, fontweight="bold")
    pad_x, pad_y = 0.06 * Lx, 0.06 * Ly
    ax_plate.set_xlim(-pad_x, Lx + pad_x)
    ax_plate.set_ylim(-pad_y, Ly + pad_y)
    ax_plate.set_aspect("equal")
    ax_plate.set_xlabel("x (m)")
    ax_plate.set_ylabel("y (m)")
    ax_plate.set_title("Both designs' resonator layouts", fontsize=11)
    ax_plate.legend(loc="upper right", fontsize=7)

fig.suptitle(
    "Dataset search: genuinely different designs producing near-identical true ERP spectra\n"
    "(nearest-neighbor search over all 100k configurations, PCA-screened + exact re-scored)",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(PAIRS_OUT, dpi=150)
plt.close(fig)
print(f"Saved {PAIRS_OUT}")

# ---------------------------------------------------------
# Figure 2: whole clusters (connected components + exact verification)
# ---------------------------------------------------------
mask = erp_mse < MSE_EDGE_THRESHOLD
pi, pj = i_idx[mask], j_idx[mask]

parent = {}
def find(x):
    while parent.get(x, x) != x:
        parent[x] = parent.get(parent[x], parent[x])
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

for a, b in zip(pi, pj):
    parent.setdefault(a, a)
    parent.setdefault(b, b)
    union(a, b)

from collections import defaultdict
comp = defaultdict(list)
for node in parent:
    comp[find(node)].append(node)

candidates = [c for c in comp.values() if 3 <= len(c) <= MAX_CLUSTER_SIZE]
candidates.sort(key=len, reverse=True)

verified = []
for c in candidates:
    sub = erp[c]
    n = len(c)
    d2 = np.empty((n, n))
    for a in range(n):
        d2[a] = ((sub - sub[a]) ** 2).mean(axis=1)
    max_mse = d2[np.triu_indices(n, 1)].max()
    mean_mse = d2[np.triu_indices(n, 1)].mean()
    verified.append((sorted(int(x) for x in c), max_mse, mean_mse))

verified.sort(key=lambda t: len(t[0]), reverse=True)
clusters_to_plot = verified[:N_CLUSTERS_TO_PLOT]
print(f"\nPlotting {len(clusters_to_plot)} verified clusters (sizes: {[len(c[0]) for c in clusters_to_plot]})")

n_rows = len(clusters_to_plot)
fig = plt.figure(figsize=(15.5, 4.6 * n_rows))
gs = fig.add_gridspec(n_rows, 3, width_ratios=[1.55, 1.0, 0.55], wspace=0.35, hspace=0.55)
letters = "ABCDEFGHIJ"

for row, (members, max_mse, mean_mse) in enumerate(clusters_to_plot):
    ax_erp = fig.add_subplot(gs[row, 0])
    ax_plate = fig.add_subplot(gs[row, 1])
    ax_legend = fig.add_subplot(gs[row, 2])

    m = len(members)
    colors = cm.tab20(np.linspace(0, 1, max(m, 2)))

    handles = []
    for k, idx in enumerate(members):
        (line,) = ax_erp.plot(freqs, erp[idx], lw=1.6, color=colors[k], alpha=0.9)
        handles.append(line)
    ax_erp.set_xlabel("Frequency (Hz)")
    ax_erp.set_ylabel("ERP (dB)")
    ax_erp.set_title(
        f"Cluster {letters[row]}: {m} designs, spectra overlaid\n"
        f"max pairwise MSE={max_mse:.3f} dB², mean={mean_mse:.3f} dB²",
        fontsize=10.5,
    )
    ax_erp.grid(alpha=0.3)

    ax_plate.add_patch(plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor="black", lw=1.5))
    ax_plate.scatter([xf], [yf], marker="*", s=150, color="black", edgecolors="white",
                      linewidths=0.6, zorder=5)
    for k, idx in enumerate(members):
        cfg = configs[idx]
        ax_plate.scatter(cfg[:, 3], cfg[:, 4], marker="o", s=65, color=colors[k],
                          edgecolors="black", linewidths=0.5, zorder=3, alpha=0.9)
    pad_x, pad_y = 0.06 * Lx, 0.06 * Ly
    ax_plate.set_xlim(-pad_x, Lx + pad_x)
    ax_plate.set_ylim(-pad_y, Ly + pad_y)
    ax_plate.set_aspect("equal")
    ax_plate.set_xlabel("x (m)")
    ax_plate.set_ylabel("y (m)")
    ax_plate.set_title("Resonator layouts (★ = force point)", fontsize=10.5)

    ax_legend.axis("off")
    labels = [f"#{idx}" for idx in members]
    ncol = 2 if m > 6 else 1
    ax_legend.legend(
        handles, labels, loc="center left", frameon=False, fontsize=8.5,
        ncol=ncol, title="Design ID", title_fontsize=9, handlelength=1.6,
        columnspacing=1.0, labelspacing=0.6,
    )

fig.suptitle(
    "Whole clusters of mutually near-identical true ERP spectra from genuinely different designs\n"
    "(same color = same design, matched between spectrum, layout, and the legend)",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(CLUSTERS_OUT, dpi=150)
plt.close(fig)
print(f"Saved {CLUSTERS_OUT}")
print("\nDone.")
