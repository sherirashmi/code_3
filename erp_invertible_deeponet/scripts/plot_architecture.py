"""Diagram: the forward DeepONet (DON) next to the invertible DeepONet (iDON, Q = 64).

(a) DON (erp_forward/scripts/don.py): set encoder -> context -> branch
    coefficients; trunk basis of frequency modulated by the context (FiLM),
    4-term inner product, residual frequency refinement. Forward only.
(b) iDON Q64 (erp_invertible_deeponet/scripts/model.py): RealNVP branch on the padded
    design, fixed QR-orthonormal trunk basis, ERP = Psi b + psi_0; the inverse
    (orange) runs through the same weights: projection, sampling, RealNVP^-1.

Parameter counts are computed from freshly built models (same configuration as
the trained ones).

Run from the repository root:  python -m erp_invertible_deeponet.scripts.plot_architecture
"""

from __future__ import annotations

import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from erp_forward.scripts.don import DEFAULT_MODEL_CONFIG, build_model
from erp_forward.scripts.neural_operator_utils import build_operator_model
from utils.paths import IDON_ROOT, architecture_dir
from utils.plotting import save_figure

from .model import InvertibleDeepONet
from .train import DATASET, plot_dir

_bank = importlib.import_module("erp_inverse.scripts.fixed_resonator.plot_bank_architecture")
arrow = _bank.arrow


def box(*args, title_size=12.5, body_size=11, **kw):
    _bank.box(*args, title_size=title_size, body_size=body_size, **kw)
BLUE, ORANGE, GREEN, GREY, PURPLE, LIGHT = _bank.BLUE, _bank.ORANGE, _bank.GREEN, _bank.GREY, _bank.PURPLE, _bank.LIGHT

N_FREQ, NUM_RES, PAD = 301, 2, 56


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    # the final trained forward DON: 3 resonators, set + f_t-sorted encoder, physical plate-mode features
    don = build_operator_model(build_model, 3, {**DEFAULT_MODEL_CONFIG, "use_sorted_branch": True,
                                                "coordinate_features": "physical"})
    idon = InvertibleDeepONet(4 * NUM_RES, n_freq=N_FREQ, pad=PAD)
    P, T = DEFAULT_MODEL_CONFIG["basis_dim"], DEFAULT_MODEL_CONFIG["num_terms"]
    ctx, rw = DEFAULT_MODEL_CONFIG["context_dim"], DEFAULT_MODEL_CONFIG["refine_width"]
    refine = count(don.refine_lift) + count(don.frequency_refinement) + count(don.refine_project)
    Q, D = idon.num_basis, idon.design_dim

    fig, ax = plt.subplots(figsize=(18, 12.5))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ======================= (a) DON =====================================================
    ax.text(0.01, 0.985, f"(a) DeepONet (DON), forward operator only: design $\\to$ ERP  "
            f"({count(don):,} parameters)", fontsize=13, weight="bold", va="top")
    yb, yt, h = 0.80, 0.625, 0.135  # branch row, trunk row, box height
    box(ax, 0.01, yb, 0.13, h, "Design", "3 resonators $\\times$\n$[m, k, f_t, x, y]$\n(unordered set)",
        LIGHT["grey"])
    box(ax, 0.17, yb, 0.15, h, "Set + sorted encoder", "+ plate mode shapes\nset: shared MLP,\nmean + max pooling\nsorted: by $f_t$, MLP\n"
        f"{count(don.configuration_encoder):,} par.", LIGHT["blue"], edge=BLUE)
    box(ax, 0.35, yb + 0.02, 0.12, h - 0.04, f"Context $c$", f"{ctx} numbers\nsummary of\nthe design",
        LIGHT["blue"], edge=BLUE)
    box(ax, 0.50, yb, 0.14, h, "Branch MLP", f"$c \\to b_{{j,p}}$\n{T} terms $\\times$ {P}\ncoefficients\n"
        f"{count(don.branch_head):,} par.", LIGHT["blue"], edge=BLUE)
    box(ax, 0.01, yt, 0.13, h, "Frequency", f"$f$ (normalised)\n{N_FREQ} points", LIGHT["grey"])
    box(ax, 0.17, yt, 0.15, h, "Trunk MLP", f"$f \\to \\varphi_{{j,p}}(f)$\n{T} $\\times$ {P} basis\nfunctions\n"
        f"{count(don.trunk):,} par.", LIGHT["green"], edge=GREEN)
    box(ax, 0.35, yt, 0.12, h, "FiLM", "$\\gamma\\,\\varphi + \\beta$\n$\\gamma = 1 + 0.25\\tanh(\\cdot)$\n"
        f"$\\gamma, \\beta$ from $c$\n{count(don.trunk_modulation):,} par.", LIGHT["green"], edge=GREEN)
    box(ax, 0.50, yt, 0.14, h, "Basis", "$\\varphi_{j,p}(f; c)$\nlinear in the trunk's\nlast hidden layer\n$\\Rightarrow$ fixed basis of\n45 functions of $f$",
        LIGHT["green"], edge=GREEN)
    box(ax, 0.67, yt + 0.04, 0.15, 0.27, "Term sum", f"per term $j$:\n$\\sum_p b_{{j,p}}\\,\\varphi_{{j,p}}(f)/\\sqrt{{{P}}}$"
        f"\n\nsoftmax weights $w_j$\n$\\sum_j w_j(\\cdot) + $ bias", LIGHT["purple"], edge=PURPLE)
    box(ax, 0.85, yt + 0.115, 0.14, 0.195, "Refinement", f"residual local\nfrequency mixer\n2 $\\times$ Conv1d (k = 3)\n"
        f"width {rw}\n{refine:,} par.", LIGHT["purple"], edge=PURPLE)
    box(ax, 0.85, yt, 0.14, 0.09, "Output", f"ERP ({N_FREQ} points)", LIGHT["grey"])
    for x0, x1 in ((0.14, 0.17), (0.32, 0.35), (0.47, 0.50)):
        arrow(ax, (x0, yb + h / 2), (x1, yb + h / 2))
        arrow(ax, (x0, yt + h / 2), (x1, yt + h / 2))
    arrow(ax, (0.41, yb + 0.02), (0.41, yt + h), color=BLUE, )
    ax.text(0.418, yb - 0.01, "$\\gamma, \\beta$", color=BLUE, fontsize=12, va="center")
    arrow(ax, (0.64, yb + h / 2), (0.67, yb + h / 2 - 0.005))
    arrow(ax, (0.64, yt + h / 2), (0.67, yt + h / 2 + 0.005))
    arrow(ax, (0.82, 0.78), (0.85, 0.78))
    arrow(ax, (0.92, yt + 0.115), (0.92, yt + 0.09))
    ax.text(0.01, 0.598, "Not invertible: the encoder is many-to-one and the refinement is nonlinear $\\Rightarrow$ no closed-form way "
            "back from an ERP to a design. Before the refinement the ERP is a combination of only 45 fixed functions of $f$.",
            fontsize=11.5, color=GREY, style="italic", va="top")
    ax.plot([0.01, 0.99], [0.565, 0.565], color=GREY, lw=0.8, ls=":")

    # ======================= (b) iDON Q64 ================================================
    ax.text(0.01, 0.55, f"(b) Invertible DeepONet (iDON, $Q = {Q}$): forward and inverse in the same weights  "
            f"({count(idon):,} parameters)", fontsize=13, weight="bold", va="top")
    yf, yi, h = 0.355, 0.165, 0.14  # forward row, inverse row
    box(ax, 0.01, yf, 0.13, h, f"Design $\\mathbf{{a}}$ ($D = {D}$)", "per resonator:\n$[m, f_t, x, y]$, sorted by $f_t$\n"
        "bounded logit coord.\n$k = m(2\\pi f_t)^2$", LIGHT["grey"])
    box(ax, 0.17, yf, 0.13, h, "Latent padding", f"$[\\mathbf{{a}}, \\mathbf{{z}}]$, $\\mathbf{{z}} = 0$\n"
        f"{D} + {PAD} = {Q} numbers\n(bijection needs\nequal sizes)", LIGHT["grey"])
    box(ax, 0.33, yi, 0.15, yf + h - yi, "RealNVP branch $T$", f"bijection $\\mathbb{{R}}^{{{Q}}} \\leftrightarrow "
        f"\\mathbb{{R}}^{{{Q}}}$\n\n{len(idon.branch.layers)} affine coupling layers\n(random-half masks)\n"
        "$s, t$: MLP 2 $\\times$ 256, SiLU\nscale clamp $e^{\\pm 1}$\n\nforward $T$ (top), inverse $T^{-1}$ (bottom)\n"
        f"exact, same weights\n{count(idon.branch):,} par.", LIGHT["blue"], edge=BLUE)
    box(ax, 0.51, yf, 0.13, h, "Coefficients $\\mathbf{b}$", f"{Q} numbers\none per basis function", LIGHT["blue"],
        edge=BLUE)
    box(ax, 0.67, yf, 0.14, h, "ERP synthesis", "$\\hat{\\mathbf{y}} = \\Psi\\,\\mathbf{b} + \\psi_0$\n"
        f"linear in $\\mathbf{{b}}$\n$\\to$ predicted ERP\n({N_FREQ} points)", LIGHT["purple"], edge=PURPLE)
    box(ax, 0.84, yi, 0.15, yf + h - yi, "Trunk (basis)", f"$f \\in [-1, 1]$, {N_FREQ} points\n\nFourier features\n"
        "$[f, \\sin k\\pi f, \\cos k\\pi f]$\n$k = 1 \\ldots 32$ (65)\n\nMLP 4 $\\times$ 256, SiLU\n"
        f"$\\to \\psi_0$ + {Q} functions\n\nQR orthonormalisation\n$\\Psi$ ({N_FREQ} $\\times$ {Q}), "
        "$\\Psi^\\top\\Psi = F\\,I$\n\nsame for every design\n" f"{count(idon.trunk):,} par.",
        LIGHT["green"], edge=GREEN)
    for x0, x1 in ((0.14, 0.17), (0.30, 0.33), (0.48, 0.51), (0.64, 0.67)):
        arrow(ax, (x0, yf + h / 2), (x1, yf + h / 2), color=BLUE)
    arrow(ax, (0.84, yf + h / 2), (0.81, yf + h / 2), color=GREEN)

    # inverse row (right to left)
    box(ax, 0.67, yi, 0.14, h, "Projection", "$\\mathbf{b}^* = \\Psi^\\top(\\mathbf{y} - \\psi_0)/F$\n"
        "closed form, exact\n(least squares)", LIGHT["orange"], edge=ORANGE)
    box(ax, 0.51, yi, 0.13, h, "Sampling", "$\\mathbf{b} \\sim \\mathcal{N}(\\mathbf{b}^*, \\frac{s^2}{F} I)$\n"
        "16 samples\n(sample 1 = $\\mathbf{b}^*$)", LIGHT["orange"], edge=ORANGE)
    box(ax, 0.17, yi, 0.13, h, "Split", f"$T^{{-1}}(\\mathbf{{b}}) = [\\hat{{\\mathbf{{a}}}}, \\hat{{\\mathbf{{z}}}}]$\n"
        f"keep $\\hat{{\\mathbf{{a}}}}$ ({D})\ndrop $\\hat{{\\mathbf{{z}}}}$ ({PAD})", LIGHT["orange"], edge=ORANGE)
    box(ax, 0.01, yi, 0.13, h, "Candidate designs", "decode $m, f_t, x, y$\n$k = m(2\\pi f_t)^2$\n"
        "16 per target ERP", LIGHT["orange"], edge=ORANGE)
    for x0, x1 in ((0.67, 0.64), (0.51, 0.48), (0.33, 0.30), (0.17, 0.14)):
        arrow(ax, (x0, yi + h / 2), (x1, yi + h / 2), color=ORANGE, ls="--")
    arrow(ax, (0.84, yi + h / 2), (0.81, yi + h / 2), color=GREEN)

    yl, hl = 0.01, 0.12
    box(ax, 0.67, yl, 0.14, hl, "Target ERP $\\mathbf{y}$", f"{N_FREQ} points\n(measured or desired)",
        LIGHT["grey"], edge=ORANGE)
    arrow(ax, (0.74, yl + hl), (0.74, yi), color=ORANGE, ls="--")
    box(ax, 0.01, yl, 0.13, hl, "Solver check", "FE solver recomputes\nthe ERP of each\ncandidate", LIGHT["grey"],
        edge=ORANGE)
    arrow(ax, (0.075, yi), (0.075, yl + hl), color=ORANGE, ls="--")
    box(ax, 0.17, yl, 0.47, hl, "Training loss (one network, both directions)",
        "forward MSE$(\\hat{\\mathbf{y}}, \\mathbf{y})$  +  $w\\,$Huber$(\\hat{\\mathbf{a}} - \\mathbf{a})$  +  "
        "$0.1\\,$Huber$(\\hat{\\mathbf{z}})$\n$\\hat{\\mathbf{a}}, \\hat{\\mathbf{z}}$ from the closed-form inverse "
        "of the training ERP;  $w$: 0 $\\to$ 1 over the first 10 epochs", LIGHT["purple"], edge=PURPLE)
    ax.text(0.84, 0.075, "Key difference to (a):\nboth use a fixed basis of $f$;\nhere it is explicit, orthonormal\nand larger "
            "($Q = 64$ vs 45),\nso the ERP is linear in $\\mathbf{b}$\n$\\Rightarrow$ exact inverse",
            fontsize=11, color=GREY, style="italic", va="center")
    ax.plot([], [], color=BLUE, lw=1.6, label="forward: design $\\to$ ERP")
    ax.plot([], [], color=ORANGE, lw=1.6, ls="--", label="inverse: ERP $\\to$ designs")
    ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.555), fontsize=10, frameon=False, ncol=2)

    path = architecture_dir(IDON_ROOT) / "don_vs_idon_architecture.png"
    save_figure(fig, path)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
