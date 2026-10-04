"""Architecture diagrams of the invertible operators iFNO, iDCO and iGNO.

Same scaffold for all three (common.py / coupling.py, after Long et al.,
arXiv:2402.11722): an architecture-specific lift P (design -> latent), an
invertible coupling stack whose gate network L is the architecture's own
layer, a readout Q (latent -> ERP); in the inverse direction a lift P'
(ERP -> latent), the SAME stack run backwards, a readout Q' (latent ->
design point estimate) and a beta-VAE for several candidate designs.
Sizes and parameter counts come from models built with the configuration of
the latest trained 2-resonator models (bounded design, bounded gate, binned
readout; see models/100k_2res_grid_18modes).

Run from the repository root:  python -m erp_invertible_operators.plot_architectures [iFNO iDCO iGNO]
"""

from __future__ import annotations

import importlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from erp_invertible_operators.registry import INVERTIBLE_OPERATORS
from utils.paths import PROJECT_ROOT
from utils.plotting import save_figure

_bank = importlib.import_module("2_res_erp_inverse_models.plot_bank_architecture")
BLUE, ORANGE, GREEN, GREY, PURPLE, LIGHT = _bank.BLUE, _bank.ORANGE, _bank.GREEN, _bank.GREY, _bank.PURPLE, _bank.LIGHT

OUT_DIR = PROJECT_ROOT / "erp_invertible_operators" / "plots" / "ARCHITECTURES"
NUM_RES, N_FREQ = 2, 301
OPTIONS = {"design_param": "bounded12", "num_res": NUM_RES, "gate": "bounded", "readout": "binned"}
EXTRA = {"iFNO": {"spectral_padding": 8}}

FWD, INV = (0.60, 0.22), (0.33, 0.22)  # rows (y, h)
COLS = [(0.01, 0.11), (0.15, 0.17), (0.35, 0.12), (0.50, 0.15), (0.68, 0.13), (0.84, 0.15)]


def count(*modules) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def rect(col, row):
    x, w = COLS[col]
    return (x, row[0], w, row[1])


def box(ax, r, title, body="", face="#ffffff", edge=GREY):
    _bank.box(ax, *r, title, body, face, edge=edge, title_size=12.5, body_size=10.5)
    return r


def right_arrow(ax, a, b, color=BLUE, ls="-"):
    _bank.arrow(ax, (a[0] + a[2], a[1] + a[3] / 2), (b[0], a[1] + a[3] / 2), color=color, ls=ls)


def left_arrow(ax, a, b, color=ORANGE, ls="--"):
    _bank.arrow(ax, (a[0], b[1] + b[3] / 2), (b[0] + b[2], b[1] + b[3] / 2), color=color, ls=ls)


def describe(name, model, cfg):
    """(lift P text, lift P' text, gate L short name, gate L detail, P params, P' params)."""
    w2 = 2 * model.width
    if name == "iFNO":
        q = cfg["query_dim"]
        return (f"set encoder $\\to c$ ({w2})\nresonance query $q(f)$ ({q})\n(as in FNO)\n"
                f"Linear $[c, q(f), f]$ $\\to$ {w2}",
                f"$[y(f),$ MLP$(f)]$ ({1 + q})\nLinear $\\to$ {w2}",
                "Fourier layer (FNO)",
                "$L(x) = \\sigma(\\mathrm{GN}(W x + K x))$\n"
                f"$K$: FFT, lowest {cfg['modes']} modes, complex weights, inverse FFT\n"
                f"(frequency axis zero-padded by {cfg.get('spectral_padding', 0)})\n"
                "$W$: Conv1d ($k = 3$), GroupNorm, GELU\nglobal + local mixing along $f$",
                count(model.configuration_encoder, model.resonance_query, model.lift_p),
                count(model.freq_embed, model.lift_pp))
    if name == "iDCO":
        return (f"branch: set encoder ({cfg['branch_dim']})\ntrunk MLP$(f)$ ({cfg['trunk_dim']})\n"
                f"resonance query $q(f)$ ({cfg['query_dim']})\n(as in DCO)\nLinear $\\to$ {w2}, SiLU",
                f"$[y(f),$ trunk$'(f)]$ ({1 + cfg['trunk_dim']})\nLinear $\\to$ {w2}, SiLU",
                "residual MLP + refinement (DCO)",
                "$L(x) = $ Refine$(\\sigma(\\mathrm{LN}(x + \\mathrm{MLP}(x))))$\n"
                "residual MLP block: pointwise in $f$ (DCO's core)\n"
                "Refine: residual 2 $\\times$ Conv1d ($k = 3$), GroupNorm\n(DCO's own last stage): local mixing along $f$",
                count(model.branch, model.trunk, model.resonance_query, model.lift_p),
                count(model.trunk_inverse, model.lift_pp))
    return (f"node lift + {cfg['depth']} $\\times$ message\npassing between resonators\n"
            "kernel + attention over\nresonators at every $f$\n(as in GNO)\n"
            f"Linear $\\to$ {w2}, SiLU",
            f"$[y(f),$ freq. enc.$(f)]$ ({1 + cfg['frequency_dim']})\nLinear $\\to$ {w2}, SiLU",
            "pointwise MLP + refinement (GNO)",
            "$L(x) = $ Refine$(\\mathrm{MLP}(x))$\n"
            "plain MLP, pointwise in $f$ (GNO's MLP style)\n"
            "Refine: residual 2 $\\times$ Conv1d ($k = 3$), GroupNorm\n(GNO's own last stage); message passing\n"
            "cannot be the gate (it acts on resonators, not on $f$)",
            count(model.node_lift, model.graph_layers, model.frequency_encoder, model.query_kernel,
                  model.attention_score, model.lift_p),
            count(model.lift_pp))


def draw(name, model, cfg):
    w, W2 = model.width, 2 * model.width
    lift_p, lift_pp, gate_short, gate_detail, p_par, pp_par = describe(name, model, cfg)
    fig, ax = plt.subplots(figsize=(18, 10.5))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    long = {"iFNO": "invertible Fourier neural operator", "iDCO": "invertible deep cat operator",
            "iGNO": "invertible graph neural operator"}[name]
    ax.text(0.01, 0.98, f"{name}: {long}  ({count(model):,} parameters)", fontsize=14, va="top")
    ax.text(0.01, 0.94, "Only the coupling stack is exactly invertible; the lifts $P, P'$ and readouts $Q, Q'$ are "
            "learned maps, so the two directions are tied together by the training losses.",
            fontsize=11, color=GREY, style="italic", va="top")

    # forward row
    D = model.design_dim
    a = box(ax, rect(0, FWD), f"Design $\\mathbf{{a}}$ ($D = {D}$)", f"{NUM_RES} res. $\\times$ $[m, f_t, x, y]$\n"
            "bounded logit coord.\n$k = m(2\\pi f_t)^2$\n$\\to$ z-scored\n$[m, k, f_t, x, y]$", LIGHT["grey"])
    p = box(ax, rect(1, FWD), "Lift $P$", f"{lift_p}\n{p_par:,} par.", LIGHT["blue"], BLUE)
    v0 = box(ax, rect(2, FWD), "Latent $v_0$", f"$(v_1, v_2)$\n2 $\\times$ {w} channels\n$\\times$ {N_FREQ} "
             "frequencies", LIGHT["blue"], BLUE)
    stack = box(ax, (COLS[3][0], INV[0], COLS[3][1], FWD[0] + FWD[1] - INV[0]), "Coupling stack",
                f"{len(model.blocks.blocks)} blocks, same weights\nboth ways\n\nforward ($\\rightarrow$):\n"
                "$v_1 \\leftarrow v_1 \\odot S(L(v_2))$\n$v_2 \\leftarrow v_2 \\odot S(L(v_1))$\n\ninverse ($\\leftarrow$):\n"
                "divide by the same gates,\nblocks in reverse order\n\n$S(x) = e^{2\\tanh(a x)}$, $a = 0$ at start\n"
                f"$L$: {gate_short}\n{count(model.blocks):,} par.", LIGHT["purple"], PURPLE)
    q = box(ax, rect(4, FWD), "Readout $Q$", f"per frequency:\nMLP {W2} $\\to$ {W2} $\\to$ 1\n"
            f"{count(model.project_q):,} par.", LIGHT["blue"], BLUE)
    y_hat = box(ax, rect(5, FWD), "Predicted ERP", f"$\\hat{{y}}(f)$, {N_FREQ} points", LIGHT["grey"])
    for s, t in ((a, p), (p, v0), (v0, stack), (q, y_hat)):
        right_arrow(ax, s, t)
    _bank.arrow(ax, (stack[0] + stack[2], FWD[0] + FWD[1] / 2), (q[0], FWD[0] + FWD[1] / 2), color=BLUE)

    # inverse row (right to left)
    y = box(ax, rect(5, INV), "Target ERP $y$", f"{N_FREQ} points\n(measured or\ndesired)", LIGHT["grey"], ORANGE)
    pp = box(ax, rect(4, INV), "Lift $P'$", f"{lift_pp}\n{pp_par:,} par.", LIGHT["orange"], ORANGE)
    bins = getattr(model, "readout_bins", 0)
    qp = box(ax, rect(2, INV), "Readout $Q'$", f"mean + max over $f$\n({4 * w})\n+ {bins} ordered bins\n"
             f"$\\to$ MLP $\\to \\hat{{\\mathbf{{a}}}}$ ({D})\n{count(model.project_qp) + count(model.readout_bin_proj):,}"
             " par.", LIGHT["orange"], ORANGE)
    vae = box(ax, rect(1, INV), "$\\beta$-VAE", f"$\\hat{{\\mathbf{{a}}}} \\to \\mu, \\sigma$ "
              f"($z$: {model.vae.z_dim})\n$z \\sim \\mathcal{{N}}(\\mu, \\sigma^2)$\ndecoder $z \\to$ design\n"
              f"{count(model.vae):,} par.", LIGHT["orange"], ORANGE)
    cand = box(ax, rect(0, INV), "Candidate\ndesigns", "several samples\nper target ERP\n$\\to$ solver check",
               LIGHT["orange"], ORANGE)
    left_arrow(ax, y, pp)
    _bank.arrow(ax, (pp[0], INV[0] + INV[1] / 2), (stack[0] + stack[2], INV[0] + INV[1] / 2), color=ORANGE, ls="--")
    for s, t in ((stack, qp), (qp, vae), (vae, cand)):
        left_arrow(ax, s, t)
    ax.text(pp[0] - 0.015, INV[0] + INV[1] / 2 + 0.012, "$u_0$", fontsize=11, color=ORANGE, ha="center")
    ax.text(stack[0] - 0.015, INV[0] + INV[1] / 2 + 0.012, "$v_0$", fontsize=11, color=ORANGE, ha="center")
    ax.text(stack[0] + stack[2] + 0.015, FWD[0] + FWD[1] / 2 + 0.012, "$v_K$", fontsize=11, color=BLUE, ha="center")

    # bottom: gate detail + training
    box(ax, (0.01, 0.03, 0.47, 0.25), f"Gate network $L$ inside every block: {gate_short}",
        f"{gate_detail}\nacts on one half ({w} channels $\\times$ {N_FREQ} frequencies); "
        f"{count(model.blocks.blocks[0].gate_net):,} par. per block", LIGHT["purple"], PURPLE)
    box(ax, (0.50, 0.03, 0.49, 0.25), "Training (3 stages)",
        "1: $J_{FWD}(\\hat y, y) + J_{INV}(\\hat{\\mathbf{a}}, \\mathbf{a}) + J_{PQ'} + J_{P'Q}$\n"
        "     + 0.1 cycle (design $\\to \\hat y \\to$ design) + 0.1 latent alignment ($P'(y) \\approx v_K$)\n"
        "2: $\\beta$-VAE on the stage-1 point estimates (reconstruction + $\\beta$ KL)\n"
        "3: all terms together, the VAE now fed by the inverse pipeline\n"
        "$J_{PQ'}$, $J_{P'Q}$: lift then readout directly (skipping the stack)",
        LIGHT["grey"], GREY)
    ax.plot([], [], color=BLUE, lw=1.6, label="forward: design $\\to$ ERP")
    ax.plot([], [], color=ORANGE, lw=1.6, ls="--", label="inverse: ERP $\\to$ designs")
    ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.99), fontsize=11, frameon=False, ncol=2)
    return fig


def main(names) -> None:
    specs = {spec["short"]: spec for spec in INVERTIBLE_OPERATORS.values()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = specs[name]
        cfg = {**spec["model_config"], **OPTIONS, **EXTRA.get(name, {})}
        model = spec["build"](4 * NUM_RES, **cfg)
        fig = draw(name, model, cfg)
        save_figure(fig, OUT_DIR / f"{name.lower()}_architecture.png")
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or [spec["short"] for spec in INVERTIBLE_OPERATORS.values()])
