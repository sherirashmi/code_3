"""Diagrams of the model bank: the Flow inside one block, and the bank of blocks.

``flow_block_architecture.png``: the conditional normalizing flow used in every
block -- ERP band (40-120 Hz) -> spectrum encoder -> conditioning embedding;
positions <-> 8 affine coupling layers <-> Gaussian latent; sizes and parameter
counts are read from a trained block checkpoint.

``block_bank_overview.png``: one column per block (f_t = 40 ... 100 Hz) with its
stiffness, data split, the block's typical ERP (median and 25-75 % band of its
configurations) and its test results in the bank, joined to the solver check
that picks the design.

Run from the repository root:  python -m erp_inverse.scripts.fixed_resonator.plot_bank_architecture [3]
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from utils.erp_dataset import DATASETS, FIXED_BLOCK_FREQUENCIES, fixed_block_tag
from utils.plotting import save_figure
from utils.support import load_dataset

from .block_bank import out_dir
from .evaluate import load_model

BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a78d6", "#eb6834", "#1f9e74", "#6f6f6a", "#8f5bb5"
LIGHT = {"blue": "#e3eefb", "orange": "#fdebe1", "green": "#e0f3ec", "grey": "#efefec", "purple": "#efe6f6"}


def box(ax, x, y, w, h, title, body="", face="#ffffff", edge=GREY, title_size=10.5, body_size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=face, ec=edge, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top", fontsize=title_size, weight="bold", zorder=3)
    if body:
        ax.text(x + w / 2, y + h / 2 - 0.018, body, ha="center", va="center", fontsize=body_size,
                linespacing=1.35, zorder=3)


def arrow(ax, p, q, color=GREY, text=None, style="-|>", ls="-", text_offset=(0.0, 0.012), rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, lw=1.4, color=color, ls=ls, zorder=4,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + text_offset[0], (p[1] + q[1]) / 2 + text_offset[1], text, ha="center",
                va="bottom", fontsize=8, color=color)


def flow_architecture(tag: str, num_res: int = 2) -> None:
    model, norm = load_model("Flow", tag)
    D = 2 * num_res
    coords = ", ".join(f"x_{r + 1}, y_{r + 1}" for r in range(num_res))
    enc = sum(p.numel() for p in model.encoder.parameters())
    per_layer = sum(p.numel() for p in model.layers[0].parameters())
    total = sum(p.numel() for p in model.parameters())
    n_layers, n_band = len(model.layers), int(norm["n_freq_band"])

    fig, ax = plt.subplots(figsize=(16, 8.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.92)
    ax.axis("off")

    # ---- conditioning path (top) ------------------------------------------------------------
    box(ax, 0.01, 0.70, 0.15, 0.2, "Input: ERP spectrum",
        f"40--120 Hz band\n{n_band} frequency points\nnormalised (z-score)", LIGHT["grey"])
    box(ax, 0.21, 0.70, 0.22, 0.2, "Spectrum encoder",
        "3 $\\times$ [Conv1d (kernel 5, 48 ch.) + SiLU]\nglobal mean + max pooling (96)\n"
        f"MLP 96 $\\to$ 96 $\\to$ 96\n{enc:,} parameters", LIGHT["blue"], edge=BLUE)
    box(ax, 0.48, 0.73, 0.14, 0.14, "Conditioning $c$", "96-d embedding of the ERP\nfed to every coupling layer",
        LIGHT["blue"], edge=BLUE)
    arrow(ax, (0.16, 0.80), (0.21, 0.80))
    arrow(ax, (0.43, 0.80), (0.48, 0.80))

    # ---- invertible path (middle) -------------------------------------------------------------
    y0, h = 0.36, 0.2
    box(ax, 0.01, y0, 0.15, h, "Positions",
        f"$[{coords}]$\nresonators ordered by $x$\nlogit-bounded to the plate,\nnormalised $\\to$ {D} numbers",
        LIGHT["orange"], edge=ORANGE)
    xs = np.linspace(0.205, 0.735, n_layers)
    w = 0.052
    for i, x in enumerate(xs):
        mask = "$x_1, x_2$ fixed" if i % 2 == 0 else "$y_1, y_2$ fixed"
        box(ax, x, y0 + 0.02, w, h - 0.04, f"C{i + 1}", "affine\ncoupling\n" + ("odd" if i % 2 else "even"),
            "#ffffff", edge=PURPLE, title_size=9.5, body_size=7.2)
        arrow(ax, (0.48 + 0.07, 0.73), (x + w / 2, y0 + h - 0.02), color=BLUE, ls=(0, (2, 2)), style="-|>")
        if i:
            arrow(ax, (xs[i - 1] + w, y0 + h / 2 + 0.03), (x, y0 + h / 2 + 0.03), color=GREY)
            arrow(ax, (x, y0 + h / 2 - 0.03), (xs[i - 1] + w, y0 + h / 2 - 0.03), color=GREEN)
    arrow(ax, (0.16, y0 + h / 2 + 0.03), (xs[0], y0 + h / 2 + 0.03), color=GREY, text="training", text_offset=(0, 0.008))
    arrow(ax, (xs[0], y0 + h / 2 - 0.03), (0.16, y0 + h / 2 - 0.03), color=GREEN, text="sampling", text_offset=(0, -0.035))
    box(ax, 0.835, y0, 0.155, h, "Latent $z$", f"{D} numbers\n$z \\sim \\mathcal{{N}}(0, I)$\n(standard Gaussian)",
        LIGHT["green"], edge=GREEN)
    arrow(ax, (xs[-1] + w, y0 + h / 2 + 0.03), (0.835, y0 + h / 2 + 0.03), color=GREY)
    arrow(ax, (0.835, y0 + h / 2 - 0.03), (xs[-1] + w, y0 + h / 2 - 0.03), color=GREEN)

    # ---- one coupling layer (detail) ------------------------------------------------------------
    box(ax, 0.20, 0.03, 0.42, 0.25, "Inside one affine coupling layer",
        f"split the {D} numbers by an alternating mask: kept half $u_a$, changed half $u_b$\n"
        f"MLP$([u, c])$: {D} + 96 $\\to$ 96 $\\to$ 96 $\\to$ {2 * D}, SiLU $\\Rightarrow$ $\\log s$, $t$\n"
        "$u_b' = u_b \\odot \\exp(\\log s) + t$, with $\\log s = 2 \\tanh(\\cdot)$ (bounded)\n"
        "exactly invertible: $u_b = (u_b' - t) \\odot \\exp(-\\log s)$\n"
        f"{per_layer:,} parameters per layer, {n_layers} layers", LIGHT["purple"], edge=PURPLE, body_size=8.6)
    arrow(ax, (xs[3] + w / 2, y0), (0.41, 0.28), color=PURPLE, ls=(0, (2, 2)))
    box(ax, 0.66, 0.03, 0.33, 0.25, "Training and use",
        "training: maximise $\\log p(\\mathrm{positions} \\mid \\mathrm{ERP})$\n"
        "= $\\log \\mathcal{N}(z) + \\sum \\log |\\det J|$ (exact likelihood)\n"
        "use: draw $z$, run the layers backwards $\\Rightarrow$ positions\n"
        "+ the block's fixed $m$, $f_t$ $\\Rightarrow$ full design\n"
        f"total {total:,} parameters per block", LIGHT["grey"], body_size=8.6)
    box(ax, 0.01, 0.03, 0.15, 0.25, "Output design",
        f"$m = {float(norm['fixed_m']):.1f}$ kg (fixed)\n$f_t$ = block value\n$k = m (2\\pi f_t)^2$\n"
        f"${coords}$\nfrom the flow", LIGHT["orange"], edge=ORANGE)
    arrow(ax, (0.085, y0), (0.085, 0.28), color=GREEN)
    fig.suptitle(f"Conditional normalizing flow inside each block of the {num_res}-resonator model bank "
                 "(the same architecture in all 7 blocks, trained separately)", fontsize=13, y=0.985)
    save_figure(fig, out_dir(num_res) / "flow_block_architecture.png")


def _summary_by_block(num_res: int = 2) -> dict[float, tuple[float, float, float]]:
    text = (out_dir(num_res) / "block_bank_summary.txt").read_text()
    rows = re.findall(r"f_t =\s+([\d.]+) Hz:\s+([\d.]+) %\s+([\d.]+) dB\s+([\d.]+) cm", text)
    return {float(f): (float(a), float(e), float(d)) for f, a, e, d in rows}


def bank_overview(num_res: int = 2) -> None:
    summary = _summary_by_block(num_res)
    B = len(FIXED_BLOCK_FREQUENCIES)
    fig = plt.figure(figsize=(17, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cmap = plt.get_cmap("viridis")

    box(ax, 0.36, 0.86, 0.28, 0.08, "Given ERP spectrum", "the 40--120 Hz band goes to EVERY block", LIGHT["grey"])
    left, width, gap = 0.02, 0.122, 0.0187
    for b, f_t in enumerate(FIXED_BLOCK_FREQUENCIES):
        x = left + b * (width + gap)
        color = cmap(b / (B - 1))
        k = 0.2 * (2 * np.pi * f_t) ** 2
        acc, err, dist = summary.get(f_t, (np.nan, np.nan, np.nan))
        ax.add_patch(FancyBboxPatch((x, 0.255), width, 0.535, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    fc="#ffffff", ec=color, lw=2.2, zorder=2))
        ax.text(x + width / 2, 0.775, f"Block {b + 1}", ha="center", va="top", fontsize=11, weight="bold")
        ax.text(x + width / 2, 0.745, f"$f_t = {f_t:g}$ Hz\n$m = 0.2$ kg, $k = {k / 1000:.1f}$ kN/m",
                ha="center", va="top", fontsize=9, linespacing=1.4)
        ax.text(x + width / 2, 0.665, "data: 10,000 configs\n8,000 / 1,000 / 1,000\n(train / val / test)",
                ha="center", va="top", fontsize=8, color=GREY, linespacing=1.3)
        ax.text(x + width / 2, 0.575, "Flow (200k parameters)", ha="center", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc=LIGHT["purple"], ec=PURPLE, lw=0.8))
        # typical ERP of the block
        data = load_dataset(DATASETS[fixed_block_tag(f_t, num_res=num_res)]["files"][0])
        freq = np.asarray(data["frequency_values"], dtype=np.float64)
        erp = np.asarray(data["responses"], dtype=np.float64)[:, :, 0]
        band = (freq >= 40) & (freq <= 120)
        q1, med, q3 = np.percentile(erp[:, band], [25, 50, 75], axis=0)
        inset = fig.add_axes([x + 0.008, 0.37, width - 0.016, 0.155])
        inset.fill_between(freq[band], q1, q3, color=color, alpha=0.3, lw=0)
        inset.plot(freq[band], med, color=color, lw=1.2)
        inset.axvline(f_t, color="#c2412c", ls="--", lw=0.9)
        inset.set_xlim(40, 120)
        inset.set_ylim(60, 125)
        inset.set_xticks([40, 80, 120])
        inset.tick_params(labelsize=6.5, length=2)
        inset.set_xlabel("$f$ (Hz)", fontsize=7, labelpad=1)
        if b == 0:
            inset.set_ylabel("ERP (dB)", fontsize=7, labelpad=1)
        else:
            inset.set_yticklabels([])
        ax.text(x + width / 2, 0.335, f"correct block: {acc:.0f} %\nERP error: {err:.2f} dB\nposition error: {dist:.1f} cm",
                ha="center", va="top", fontsize=8.2, linespacing=1.35)
        arrow(ax, (0.5, 0.86), (x + width / 2, 0.792), color=GREY)
        arrow(ax, (x + width / 2, 0.255), (0.5, 0.165), color=color)

    box(ax, 0.30, 0.075, 0.40, 0.09, "Solver check",
        "every block's 16--32 candidates $\\to$ solver ERP $\\to$ error vs the given ERP (40--120 Hz)\n"
        "lowest error wins; best error $> 3$ dB $\\Rightarrow$ no block fits", LIGHT["green"], edge=GREEN, body_size=8.6)
    box(ax, 0.74, 0.075, 0.24, 0.09, "Output", "$f_t$ of the winning block + positions\n(plus close alternatives)",
        LIGHT["orange"], edge=ORANGE, body_size=8.6)
    arrow(ax, (0.70, 0.12), (0.74, 0.12), color=GREEN)
    ax.text(0.02, 0.045, "Insets: median (line) and 25--75 % range (band) of the ERP over the block's 10,000 configurations; "
            "dashed line: the block's $f_t$. Test results: 100 test targets per block through the whole bank.",
            fontsize=8, color=GREY, va="top")
    ax.text(0.5, 0.993, f"Model bank: one block per tuning frequency ({num_res} identical resonators, $m = 0.2$ kg)",
            ha="center", va="top", fontsize=14)
    save_figure(fig, out_dir(num_res) / "block_bank_overview.png")


def main(num_res: int = 2) -> None:
    flow_architecture(fixed_block_tag(70.0, num_res=num_res), num_res)
    bank_overview(num_res)


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
