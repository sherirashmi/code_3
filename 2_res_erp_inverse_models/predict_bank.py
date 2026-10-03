"""Inverse design with the trained model bank: ERP spectrum in, configurations out.

The ERP goes through every block (f_t = 40, 50, ..., 100 Hz, m = 0.2 kg, 2
resonators); each block samples position sets, every candidate is checked with
the solver, and the candidates with the lowest 40-120 Hz ERP error are returned
(the best one plus distinct alternatives, e.g. y-mirror images).

Input ERP (dB) -- one of:

* ``--erp FILE.csv``: two columns ``frequency_hz, erp_db`` (header optional), or a
  single column of 301 values on the dataset grid (10-160 Hz, 0.5 Hz steps);
* ``--erp FILE.npy``: shape (301,) on the dataset grid, or (2, n) = frequency, ERP;
* ``--config F_T X1 Y1 X2 Y2``: a configuration whose ERP is computed with the
  solver first (for trying the bank; the true answer is then shown too).

Any frequency grid covering 40-120 Hz works (it is interpolated). The physics
must match the bank: same plate, excitation force and 2 resonators of 0.2 kg.

Usage (from the repository root)::

    python -m 2_res_erp_inverse_models.predict_bank --config 70 0.40 0.15 1.10 0.35
    python -m 2_res_erp_inverse_models.predict_bank --erp my_erp.csv --top 3

Saves ``plots/block_bank/predictions/<name>.png`` and ``<name>.txt``.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from erp_inverse_operators.evaluate import SPAWN_CONTEXT, solve_configs
from utils.physics import Ly, freqs
from utils.plotting import ERP_LABEL, FREQ_LABEL, save_figure

from .block_bank import OUT_DIR, BlockBank
from .design_space import sort_by_x
from .evaluate import RES_COLORS, _draw_plate

NO_FIT_DB = 3.0  # best band error above this: no block reproduces the ERP well
CANDIDATE_COLORS = ("#eb6834", "#2a78d6", "#1f9e74", "#9467bd", "#8c564b")


def load_erp(path: str | Path, grid: np.ndarray) -> np.ndarray:
    """ERP (dB) from .csv / .npy, interpolated onto ``grid``."""
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path)
    else:
        try:
            data = np.loadtxt(path, delimiter=",")
        except ValueError:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 2 and data.shape[0] != 2:
        data = data.T  # columns -> rows
    if data.ndim == 1 or data.shape[0] == 1:
        values = data.ravel()
        if values.size != grid.size:
            raise ValueError(f"A single ERP column needs {grid.size} values on the dataset grid "
                             f"({grid[0]:g}-{grid[-1]:g} Hz); got {values.size}. Give frequency, ERP columns instead.")
        return values
    f, erp = data[0], data[1]
    if f.min() > 40.0 or f.max() < 120.0:
        raise ValueError(f"The ERP must cover 40-120 Hz (given: {f.min():g}-{f.max():g} Hz).")
    return np.interp(grid, f, erp, left=np.nan, right=np.nan)


def design_from_config(f_t: float, x1: float, y1: float, x2: float, y2: float, m: float = 0.2) -> np.ndarray:
    k = m * (2 * np.pi * f_t) ** 2
    return sort_by_x(np.array([[m, k, f_t, x1, y1], [m, k, f_t, x2, y2]], dtype=np.float64))


def rank_candidates(res: dict, top: int, min_distinct_cm: float = 3.0):
    """Best candidates over all blocks and samples, skipping near-duplicates of better ones."""
    designs, rmse = res["designs"][0], res["band_rmse"][0]  # (B, S, R, 5), (B, S)
    order = np.dstack(np.unravel_index(np.argsort(rmse, axis=None), rmse.shape))[0]
    picked = []
    for b, s in order:
        d = designs[b, s]
        if any(abs(d[0, 2] - p[0, 2]) < 1e-6 and
               np.all(100 * np.hypot(d[:, 3] - p[:, 3], d[:, 4] - p[:, 4]) < min_distinct_cm) for p, _, _ in picked):
            continue
        picked.append((d, int(b), float(rmse[b, s])))
        if len(picked) == top:
            break
    return picked


def main(erp_db=None, true_design=None, name: str = "prediction", top: int = 3, num_samples: int = 32):
    grid = np.asarray(freqs, dtype=np.float64)
    bank = BlockBank()
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT) as pool:
        if erp_db is None:
            erp_db = solve_configs(pool, true_design[None], grid)[0]
        if np.isnan(erp_db).any():  # outside the given grid: keep only what the bank uses
            erp_db = np.where(np.isnan(erp_db), np.nanmean(erp_db), erp_db)
        res = bank.predict(erp_db[None], grid, pool, num_samples)
        candidates = rank_candidates(res, top)
        solved = solve_configs(pool, np.stack([c[0] for c in candidates]), grid)
    mask = res["mask"]
    best_per_block = res["best_per_block"][0]
    best_err = candidates[0][2]

    lines = [f"Model bank prediction ({len(bank.tags)} blocks, f_t = {', '.join(f'{f:g}' for f in bank.block_ft)} Hz, "
             f"m = 0.2 kg; {num_samples} samples per block, solver-checked)", ""]
    if best_err > NO_FIT_DB:
        lines += [f"WARNING: best ERP error {best_err:.2f} dB > {NO_FIT_DB:g} dB -- no block reproduces this ERP well "
                  "(f_t between the blocks, a different mass, or different physics).", ""]
    if true_design is not None:
        lines.append("True configuration: f_t = {:g} Hz; ".format(true_design[0, 2]) + "; ".join(
            f"R{r + 1} ({true_design[r, 3]:.3f}, {true_design[r, 4]:.3f}) m" for r in range(len(true_design))))
        lines.append("")
    for rank, (d, b, err) in enumerate(candidates, 1):
        pos = "; ".join(f"R{r + 1} ({d[r, 3]:.3f}, {d[r, 4]:.3f}) m" for r in range(len(d)))
        extra = ""
        if true_design is not None:
            dist = 100 * np.hypot(d[:, 3] - true_design[:, 3], d[:, 4] - true_design[:, 4])
            mirror = true_design.copy()
            mirror[:, 4] = Ly - mirror[:, 4]
            dist_m = 100 * np.hypot(d[:, 3] - mirror[:, 3], d[:, 4] - mirror[:, 4])
            extra = f" | position error {', '.join(f'{v:.1f}' for v in dist)} cm (to y-mirror: {', '.join(f'{v:.1f}' for v in dist_m)} cm)"
        lines.append(f"{rank}. f_t = {bank.block_ft[b]:g} Hz, m = 0.2 kg, k = {d[0, 1]:,.0f} N/m; {pos} | "
                     f"ERP error (40-120 Hz) {err:.2f} dB{extra}")
    lines += ["", "Best ERP error of every block: " + ", ".join(f"{f:g} Hz: {e:.2f} dB" for f, e in zip(bank.block_ft, best_per_block))]
    text = "\n".join(lines)
    out_dir = OUT_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.txt").write_text(text + "\n")
    print(text)

    # ---- figure -------------------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], hspace=0.42, wspace=0.22)
    ax_bar = fig.add_subplot(gs[0, 0])
    win = candidates[0][1]
    ax_bar.bar([f"{f:g}" for f in bank.block_ft], best_per_block,
               color=["#eb6834" if b == win else "#9a9a94" for b in range(len(bank.block_ft))])
    ax_bar.axhline(NO_FIT_DB, color="#c2412c", ls="--", lw=1.0, label=f"Good-fit limit ({NO_FIT_DB:g} dB)")
    ax_bar.set_xlabel("Block, $f_t$ (Hz)")
    ax_bar.set_ylabel("Best ERP error, 40--120 Hz (dB)")
    ax_bar.set_title("Best candidate of every block (orange: chosen)")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(fontsize=9)

    ax_plate = fig.add_subplot(gs[0, 1])
    _draw_plate(ax_plate, true_design if true_design is not None else candidates[0][0][:0])
    for rank, (d, b, err) in enumerate(candidates):
        for r in range(len(d)):
            ax_plate.plot(d[r, 3], d[r, 4], "X", color="white", mec=CANDIDATE_COLORS[rank], mew=2.0, ms=11 - 1.5 * rank,
                          ls="", zorder=7, label=f"Design {rank + 1} ($f_t$ = {bank.block_ft[b]:g} Hz)" if r == 0 else None)
    ax_plate.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)
    ax_plate.set_title("Predicted positions" + (" (filled: true positions)" if true_design is not None else ""))

    ax = fig.add_subplot(gs[1, :])
    ax.axvspan(grid[mask][0], grid[mask][-1], color="#e8eef8", zorder=0, label="ERP band used (40--120 Hz)")
    ax.plot(grid, erp_db, color="black", lw=2.2, zorder=4, label="Given ERP")
    for rank, ((d, b, err), spec) in enumerate(zip(candidates, solved)):
        ax.plot(grid, spec, color=CANDIDATE_COLORS[rank], lw=1.5 if rank == 0 else 1.1, ls="--" if rank == 0 else ":",
                zorder=3, label=f"Design {rank + 1}: $f_t$ = {bank.block_ft[b]:g} Hz (error {err:.2f} dB)")
    ax.set_xlim(grid.min(), grid.max())
    ax.set_xlabel(FREQ_LABEL)
    ax.set_ylabel(ERP_LABEL)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    title = f"Model bank: designs for the given ERP (best: $f_t$ = {bank.block_ft[win]:g} Hz, error {best_err:.2f} dB)"
    if best_err > NO_FIT_DB:
        title += " -- no block fits well"
    fig.suptitle(title, fontsize=13)
    save_figure(fig, out_dir / f"{name}.png")
    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--erp", help="ERP file (.csv or .npy), see above")
    group.add_argument("--config", nargs=5, type=float, metavar=("F_T", "X1", "Y1", "X2", "Y2"),
                       help="compute the ERP of this configuration with the solver first")
    parser.add_argument("--top", type=int, default=3, help="number of designs to return (default 3)")
    parser.add_argument("--samples", type=int, default=32, help="samples per block (default 32)")
    parser.add_argument("--name", default=None, help="output file name (default from the input)")
    args = parser.parse_args()
    if args.config:
        design = design_from_config(*args.config)
        name = args.name or "config_ft{:g}_{:g}_{:g}_{:g}_{:g}".format(*args.config)
        main(true_design=design, name=name, top=args.top, num_samples=args.samples)
    else:
        erp = load_erp(args.erp, np.asarray(freqs, dtype=np.float64))
        main(erp_db=erp, name=args.name or Path(args.erp).stem, top=args.top, num_samples=args.samples)
