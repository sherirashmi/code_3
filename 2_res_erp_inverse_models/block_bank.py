"""Model bank: one position-only inverse model per fixed (m, f_t) block.

Blocks: 2 identical resonators, m = 0.2 kg, f_t = 40, 50, ..., 100 Hz (datasets
``10k_2res_fixed_m0.2_ft<f_t>_18modes``), one Flow trained per block on its own
80 % training split (``train_all``).

Inverse design with the bank, for a target ERP:

1. the ERP goes through EVERY block: each block normalises it with its own
   statistics and samples ``num_samples`` position sets, assuming its own m, f_t;
2. every suggestion (block m, f_t + sampled positions) is run through the actual
   solver and scored by its ERP error against the target in the 40-120 Hz band;
3. the lowest-error design wins -- its block gives f_t, its sample the positions.
   The best error of every block is kept, so close runners-up (alternatives)
   and "no block fits" (best error still large) are visible.

The block models' own log-densities are NOT used to choose between blocks: each
model has only seen its own f_t, so densities of different blocks are not on a
common scale and an out-of-block ERP can still get a confident score.

Evaluation (``python -m 2_res_erp_inverse_models.block_bank``): ``per_block``
targets from the TEST split of every block (never seen in training), plus
targets with f_t between the blocks (45, 75 Hz). Saves to
``plots/block_bank/``: block-selection confusion matrix, the matrix of median
best errors (true block x candidate block), error distributions, example
targets, and ``block_bank_summary.txt``.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_forward_operators.neural_operator_utils import _configuration_features, _split_ids
from erp_inverse_operators.evaluate import SPAWN_CONTEXT, solve_configs
from utils.erp_dataset import FIXED_BLOCK_FREQUENCIES, fixed_block_tag, normalize_erp_array, select_dataset_modal_resolution
from utils.physics import Ly
from utils.plotting import ERP_LABEL, FREQ_LABEL, save_figure
from utils.support import device, lhs_sampling

from .common import PACKAGE_ROOT, band_mask, model_path, prepare_data
from .design_space import sort_by_x
from .evaluate import RES_COLORS, _draw_plate, load_model, sample_designs

MODEL = "Flow"
OUT_DIR = PACKAGE_ROOT / "plots" / "block_bank"  # 2 resonators; see out_dir()


def out_dir(num_res: int = 2):
    """plots/block_bank (2 resonators) or plots/block_bank_<N>res."""
    return OUT_DIR if int(num_res) == 2 else PACKAGE_ROOT / "plots" / f"block_bank_{int(num_res)}res"
BETWEEN_BLOCKS_FT = (45.0, 75.0)


class BlockBank:
    """All trained blocks; ``predict`` sends target ERPs through every block."""

    def __init__(self, frequencies=FIXED_BLOCK_FREQUENCIES, model: str = MODEL, num_res: int = 2):
        self.num_res = int(num_res)
        tags = [fixed_block_tag(f, num_res=self.num_res) for f in frequencies]
        self.tags = [t for t in tags if model_path(model, t).exists()]
        if not self.tags:
            raise FileNotFoundError("No trained blocks found -- run train_all for each block dataset first.")
        select_dataset_modal_resolution(self.tags[0])
        self.blocks = [load_model(model, tag) for tag in self.tags]
        self.block_ft = np.array([float(norm["fixed_f_t"]) for _, norm in self.blocks])
        self.model = model

    def predict(self, true_erp_db: np.ndarray, frequency_values: np.ndarray, pool, num_samples: int = 16):
        """``true_erp_db``: (n, n_freq) target ERPs in dB on ``frequency_values``.

        Returns a dict with ``designs`` (n, B, S, R, 5), ``band_rmse`` (n, B, S),
        ``best_per_block`` (n, B) and ``choice`` (n,) = index of the winning block.
        """
        n = true_erp_db.shape[0]
        designs, rmse = [], []
        for model, norm in self.blocks:
            mask = band_mask(frequency_values, float(norm["f_low"]), float(norm["f_high"]))
            spectrum = torch.from_numpy(normalize_erp_array(true_erp_db, norm)[:, mask]).float().to(device)
            samples, _ = sample_designs(model, spectrum, num_samples, norm)  # (n, S, R, 5)
            num_res = samples.shape[2]
            solved = solve_configs(pool, samples.reshape(n * num_samples, num_res, 5), frequency_values)
            solved = solved.reshape(n, num_samples, -1)
            rmse.append(np.sqrt(((solved[..., mask] - true_erp_db[:, None, mask]) ** 2).mean(-1)))
            designs.append(samples)
        designs = np.stack(designs, axis=1)
        rmse = np.stack(rmse, axis=1)
        best_per_block = rmse.min(-1)
        return {"designs": designs, "band_rmse": rmse, "best_per_block": best_per_block,
                "choice": best_per_block.argmin(-1), "mask": mask}


def _test_targets(tags, per_block: int):
    """``per_block`` configurations from every block's TEST split: designs (sorted by x),
    ERPs, true block index, frequency grid."""
    designs, erps, labels = [], [], []
    for b, tag in enumerate(tags):
        dataset, _ = prepare_data(tag)
        ids = _split_ids(dataset)["test"][:per_block]
        designs.append(sort_by_x(_configuration_features(dataset)[ids]))
        erps.append(np.asarray(dataset.responses, dtype=np.float64)[ids, :, 0])
        labels.append(np.full(len(ids), b))
    return np.concatenate(designs), np.concatenate(erps), np.concatenate(labels), np.asarray(dataset.frequency_values, dtype=np.float64)


def _between_block_targets(pool, frequency_values, num: int = 40, mass: float = 0.2, num_res: int = 2):
    from utils.physics import resonator_bounds

    out = {}
    for f_t in BETWEEN_BLOCKS_FT:
        raw = lhs_sampling(num, bounds=resonator_bounds(num_res), seed=int(f_t)).reshape(num, num_res, 4)
        designs = np.zeros((num, num_res, 5))
        designs[..., 0], designs[..., 2] = mass, f_t
        designs[..., 1] = mass * (2 * np.pi * f_t) ** 2
        designs[..., 3:5] = raw[..., 0:2]
        designs = sort_by_x(designs)
        out[f_t] = (designs, solve_configs(pool, designs, frequency_values))
    return out


def _distance_cm(pred, true):
    return 100 * np.hypot(pred[..., 3] - true[..., 3], pred[..., 4] - true[..., 4])


def main(per_block: int = 100, num_samples: int = 16, num_examples: int = 3, num_res: int = 2) -> None:
    bank = BlockBank(num_res=num_res)
    OUT_DIR = out_dir(num_res)
    B = len(bank.tags)
    designs, erps, labels, freq = _test_targets(bank.tags, per_block)
    n = len(labels)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Bank of {B} blocks (f_t = {', '.join(f'{f:g}' for f in bank.block_ft)} Hz); "
          f"{n} test targets x {B} blocks x {num_samples} samples, every design checked with the solver ...")

    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT) as pool:
        res = bank.predict(erps, freq, pool, num_samples)
        between = _between_block_targets(pool, freq, num_res=num_res)
        between_res = {f: bank.predict(e, freq, pool, num_samples) for f, (_, e) in between.items()}

    choice, best = res["choice"], res["best_per_block"]
    rows = np.arange(n)
    correct = choice == labels
    sorted_best = np.sort(best, axis=1)
    margin = sorted_best[:, 1] - sorted_best[:, 0]
    chosen = res["designs"][rows, choice, res["band_rmse"][rows, choice].argmin(-1)]  # (n, R, 5)
    oracle = res["designs"][rows, labels, res["band_rmse"][rows, labels].argmin(-1)]
    dist_chosen, dist_oracle = _distance_cm(chosen, designs), _distance_cm(oracle, designs)
    mirror = designs.copy()
    mirror[..., 4] = Ly - mirror[..., 4]
    dist_mirror_ok = np.minimum(dist_chosen, _distance_cm(chosen, mirror))

    lines = [
        f"Model bank: {B} blocks ({MODEL} per block; {num_res} resonators, m = 0.2 kg; f_t = {', '.join(f'{f:g}' for f in bank.block_ft)} Hz)",
        f"{n} test targets ({per_block} from the TEST split of every block), {num_samples} samples per block,",
        "every candidate design checked with the solver; the lowest 40-120 Hz ERP error wins.", "",
        f"Correct block chosen: {correct.mean() * 100:.1f} % ({correct.sum()} / {n})",
        f"  wrong blocks are off by: " + ", ".join(f"{d:g} Hz: {c}" for d, c in zip(*np.unique(
            np.abs(bank.block_ft[choice[~correct]] - bank.block_ft[labels[~correct]]), return_counts=True))) if (~correct).any() else "  (none wrong)",
        f"Winning design's band ERP error: median {np.median(best[rows, choice]):.2f} dB, "
        f"95th percentile {np.percentile(best[rows, choice], 95):.2f} dB",
        f"Gap to the second-best block: median {np.median(margin):.2f} dB "
        f"(below 0.5 dB for {np.mean(margin < 0.5) * 100:.1f} % of targets)",
        f"Position error of the winning design: median {np.median(dist_chosen):.1f} cm, mean {dist_chosen.mean():.1f} cm "
        f"(counting y-mirror images as correct: median {np.median(dist_mirror_ok):.1f} cm, mean {dist_mirror_ok.mean():.1f} cm)",
        f"Position error if the true block were known: median {np.median(dist_oracle):.1f} cm, mean {dist_oracle.mean():.1f} cm", "",
        "Per true block: accuracy, winning band error (median), position error (median)",
    ]
    for b in range(B):
        sel = labels == b
        lines.append(f"  f_t = {bank.block_ft[b]:5.0f} Hz: {correct[sel].mean() * 100:5.1f} %  "
                     f"{np.median(best[sel, choice[sel]]):5.2f} dB  {np.median(dist_chosen[sel]):5.1f} cm")
    lines += ["", "Targets with f_t BETWEEN the blocks (no block matches exactly):"]
    for f_t, r in between_res.items():
        won = bank.block_ft[r["choice"]]
        vals, counts = np.unique(won, return_counts=True)
        lines.append(f"  f_t = {f_t:g} Hz: chosen blocks " + ", ".join(f"{v:g} Hz x{c}" for v, c in zip(vals, counts))
                     + f"; winning band error median {np.median(r['best_per_block'].min(1)):.2f} dB "
                     f"(in-block targets: {np.median(best[rows, choice]):.2f} dB)")
    summary = "\n".join(lines)
    (OUT_DIR / "block_bank_summary.txt").write_text(summary + "\n")
    print("\n" + summary)

    labels_hz = [f"{f:g}" for f in bank.block_ft]
    # ---- confusion matrix + median best error per (true block, candidate block) --------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    conf = np.zeros((B, B))
    for t, c in zip(labels, choice):
        conf[t, c] += 1
    conf = conf / conf.sum(1, keepdims=True) * 100
    im = axes[0].imshow(conf, cmap="Blues", vmin=0, vmax=100)
    for i in range(B):
        for j in range(B):
            if conf[i, j] > 0:
                axes[0].text(j, i, f"{conf[i, j]:.0f}", ha="center", va="center", fontsize=9,
                             color="white" if conf[i, j] > 60 else "black")
    axes[0].set_title(f"Chosen block (% of targets); correct: {correct.mean() * 100:.1f} %")
    fig.colorbar(im, ax=axes[0], fraction=0.046, label="Targets (%)")
    med = np.stack([np.median(best[labels == b], axis=0) for b in range(B)])
    im = axes[1].imshow(med, cmap="magma_r")
    for i in range(B):
        for j in range(B):
            axes[1].text(j, i, f"{med[i, j]:.1f}", ha="center", va="center", fontsize=8,
                         color="white" if med[i, j] > med.max() * 0.55 else "black")
    axes[1].set_title("Median best ERP error of each candidate block (dB, 40--120 Hz)")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="Band RMSE (dB)")
    for ax in axes:
        ax.set_xticks(range(B), labels_hz)
        ax.set_yticks(range(B), labels_hz)
        ax.set_xlabel("Candidate / chosen block, $f_t$ (Hz)")
        ax.set_ylabel("True block of the target, $f_t$ (Hz)")
    fig.suptitle(f"Model bank of {B} blocks ({num_res} resonators): block selection by solver on {n} test targets", fontsize=13)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "block_selection.png")

    # ---- error distributions ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].hist(best[rows, choice], bins=40, color="#2a78d6", alpha=0.85, label="Winning block")
    axes[0].hist(sorted_best[:, 1], bins=40, histtype="step", lw=1.8, color="#eb6834", label="Second-best block")
    axes[0].set_xlabel("Best ERP error of the block, 40--120 Hz (dB)")
    axes[0].set_ylabel("Number of targets")
    axes[0].legend()
    axes[1].hist(margin, bins=40, color="#1f9e74")
    axes[1].set_xlabel("Gap between the best and second-best block (dB)")
    axes[1].set_ylabel("Number of targets")
    bins = np.linspace(0, 60, 41)
    axes[2].hist(dist_chosen.ravel(), bins=bins, histtype="step", lw=1.8, color="#2a78d6",
                 label=f"Bank (block chosen by solver): median {np.median(dist_chosen):.1f} cm")
    axes[2].hist(dist_oracle.ravel(), bins=bins, histtype="step", lw=1.8, ls="--", color="black",
                 label=f"True block known: median {np.median(dist_oracle):.1f} cm")
    axes[2].set_xlabel("Position error of a resonator (cm)")
    axes[2].set_ylabel("Number of resonators")
    axes[2].legend(fontsize=8.5)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Model bank: ERP error of the winning and second-best block, their gap, and position error", fontsize=12)
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "block_bank_errors.png")

    # ---- examples: one target per figure ---------------------------------------------------
    rng = np.random.default_rng(3)
    for k, i in enumerate(rng.choice(n, num_examples, replace=False)):
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.15], hspace=0.45, wspace=0.22)
        ax_bar = fig.add_subplot(gs[0, 0])
        colors = ["#eb6834" if b == choice[i] else "#9a9a94" for b in range(B)]
        ax_bar.bar(labels_hz, best[i], color=colors)
        ax_bar.axvline(labels[i], color="black", ls=":", lw=1.2)
        ax_bar.text(labels[i], best[i].max() * 0.95, " true block", fontsize=9)
        ax_bar.set_xlabel("Block, $f_t$ (Hz)")
        ax_bar.set_ylabel("Best ERP error, 40--120 Hz (dB)")
        ax_bar.set_title("Best candidate of every block (orange: chosen)")
        ax_bar.grid(axis="y", alpha=0.3)
        ax_plate = fig.add_subplot(gs[0, 1])
        _draw_plate(ax_plate, designs[i], picked=chosen[i], picked_label="Bank")
        ax_plate.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8)
        ax_plate.set_title("True positions (filled) and the bank's prediction (crosses)")
        ax = fig.add_subplot(gs[1, :])
        mask = res["mask"]
        ax.axvspan(freq[mask][0], freq[mask][-1], color="#e8eef8", zorder=0, label="ERP band used (40--120 Hz)")
        ax.plot(freq, erps[i], color="black", lw=2.0, label=f"True ERP (block $f_t$ = {bank.block_ft[labels[i]]:g} Hz)")
        with ProcessPoolExecutor(max_workers=2, mp_context=SPAWN_CONTEXT) as pool:
            pred_erp = solve_configs(pool, chosen[i][None], freq)[0]
        ax.plot(freq, pred_erp, color="#eb6834", lw=1.5, ls="--",
                label=f"Bank's design (block {bank.block_ft[choice[i]]:g} Hz, error {best[i, choice[i]]:.2f} dB)")
        ax.set_xlim(freq.min(), freq.max())
        ax.set_xlabel(FREQ_LABEL)
        ax.set_ylabel(ERP_LABEL)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        fig.suptitle(f"Model bank, test target {k + 1}: true $f_t$ = {bank.block_ft[labels[i]]:g} Hz, "
                     f"chosen block {bank.block_ft[choice[i]]:g} Hz", fontsize=13)
        save_figure(fig, OUT_DIR / f"block_bank_example_{k + 1}.png")


if __name__ == "__main__":
    import sys

    main(num_res=int(sys.argv[1]) if len(sys.argv) > 1 else 2)
