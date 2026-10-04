"""Architecture diagrams of the forward operators (design + frequency -> ERP).

One figure per architecture, all in the same layout: inputs on the left
(design: the resonators' [m, k, f_t, x, y]; the query frequency f), the
encoders, the architecture's own core, and the output head. Box sizes and
parameter counts are read from models built with each operator's
DEFAULT_MODEL_CONFIG (the trained GENERAL/100k models, 3 resonators). The
optional f_t-sorted branch / physical-feature variants change only the
encoders' inputs, not the layout, and are noted in the footer.

Run from the repository root:  python -m erp_forward_operators.plot_architectures [DON FNO ...]
"""

from __future__ import annotations

import importlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from erp_forward_operators.neural_operator_utils import SetAndSortedResonatorEncoder, build_operator_model
from erp_forward_operators.operator_registry import OPERATORS, forward_variant
from utils.plotting import save_figure
from utils.paths import PROJECT_ROOT

_bank = importlib.import_module("2_res_erp_inverse_models.plot_bank_architecture")
BLUE, ORANGE, GREEN, GREY, PURPLE, LIGHT = _bank.BLUE, _bank.ORANGE, _bank.GREEN, _bank.GREY, _bank.PURPLE, _bank.LIGHT

OUT_DIR = PROJECT_ROOT / "erp_forward_operators" / "plots" / "ARCHITECTURES"
NUM_RES, N_FREQ = 3, 301
FINAL_VARIANT = {"sorted": True, "physical": True, "permutation": True}

# layout: rows (y, h), tall span, columns (x, w)
A, B, C = (0.64, 0.22), (0.38, 0.22), (0.12, 0.22)
TALL = (0.14, 0.70)
COLS = [(0.01, 0.11), (0.15, 0.16), (0.35, 0.14), (0.53, 0.17), (0.74, 0.12), (0.89, 0.10)]


def count(*modules) -> int:
    return sum(p.numel() for m in modules for p in m.parameters())


def rect(col, row, span=1):
    """(x, y, w, h) of a box in column ``col`` (or a (x, w) tuple), row ``row`` ((y, h) tuple)."""
    x, w = COLS[col] if isinstance(col, int) else col
    if span > 1:
        w = COLS[col + span - 1][0] + COLS[col + span - 1][1] - x
    return (x, row[0], w, row[1])


def box(ax, r, title, body="", face="#ffffff", edge=GREY):
    _bank.box(ax, *r, title, body, face, edge=edge, title_size=12.5, body_size=10.5)
    return r


def link(ax, a, b, color=GREY, ls="-", text=None):
    """Arrow from the right edge of box ``a`` to the left edge of box ``b`` (heights clamped into ``b``)."""
    p = (a[0] + a[2], a[1] + a[3] / 2)
    q = (b[0], min(max(p[1], b[1] + 0.03), b[1] + b[3] - 0.03))
    _bank.arrow(ax, p, q, color=color, ls=ls, text=text)


def condition(ax, a, b, label):
    """Curved conditioning arrow from the top of box ``a`` to the top of box ``b``."""
    p = (a[0] + 0.75 * a[2], a[1] + a[3])
    q = (b[0] + 0.5 * b[2], b[1] + b[3])
    _bank.arrow(ax, p, q, color=BLUE, ls="--", rad=-0.25)
    ax.text((p[0] + q[0]) / 2, max(p[1], q[1]) + 0.045, label, ha="center", fontsize=10.5, color=BLUE)


# ---- shared components ------------------------------------------------------------------------
def design_box(ax, r=None, body=None):
    return box(ax, r or rect(0, A), "Design", body or f"{NUM_RES} resonators\n$\\times\\,[m, k, f_t, x, y]$\n"
               "(z-scored,\nunordered set)", LIGHT["grey"])


def freq_box(ax, r=None):
    return box(ax, r or rect(0, C), "Frequency $f$", f"{N_FREQ} points\n10--160 Hz\n(z-scored)", LIGHT["grey"])


def set_encoder(ax, module, out_dim, r=None, title=None):
    """Set encoder, or (final models) set + f_t-sorted encoder fused by a Linear layer."""
    if isinstance(module, SetAndSortedResonatorEncoder):
        return box(ax, r or rect(1, A), title or "Set + sorted encoder",
                   "per resonator: $[m, k, f_t, x, y]$\n+ 120 plate mode shapes\n"
                   "set: shared MLP, mean + max\nsorted: order by $f_t$, concat., MLP\n"
                   f"Linear fusion $\\to$ context $c$ ({out_dim})\n{count(module):,} par.", LIGHT["blue"], BLUE)
    return box(ax, r or rect(1, A), title or "Set encoder", "per resonator: $[m, k, f_t, x, y]$\n"
               "+ 120 plate mode shapes\nshared MLP (tanh)\nmean + max pooling, MLP\n"
               f"$\\to$ context $c$ ({out_dim})  [{count(module):,} par.]", LIGHT["blue"], BLUE)


def query_encoder(ax, module, out_dim, r=None):
    return box(ax, r or rect(1, B), "Resonance query", "per resonator and $f$: features,\n"
               "$f$, $\\delta, |\\delta|, \\delta^2$  ($\\delta = \\frac{f - f_t}{\\sigma_f}$)\nshared MLP (tanh),\nmean + max over resonators\n"
               f"$\\to q(f)$ ({out_dim})  [{count(module):,} par.]", LIGHT["green"], GREEN)


def refinement(ax, module, width, r=None):
    return box(ax, r or rect(4, (0.35, 0.3)), "Refinement", "residual local\nfrequency mixer\n"
               f"2 $\\times$ Conv1d ($k = 3$)\nSiLU, GroupNorm\nwidth {width}\n{count(module):,} par.",
               LIGHT["purple"], PURPLE)


def output_box(ax, dims, module=None, r=None):
    par = f"\n{count(module):,} par." if module is not None else ""
    return box(ax, r or rect(5, (0.35, 0.3)), "Output head", f"{dims}\n$\\to$ ERP$(f)$\n{N_FREQ} points{par}",
               LIGHT["grey"])


def setup(title, model, footer, mode_note=True):
    fig, ax = plt.subplots(figsize=(18, 8.2))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.975, f"{title}  ({count(model):,} parameters)", fontsize=14, va="top")
    if mode_note:
        ax.text(0.01, 0.075, "Plate mode shapes: $\\sin(m\\pi x/L_x)$, $\\sin(n\\pi y/L_y)$ and their products "
                "at each resonator position.", fontsize=10.5, color=GREY, va="center")
    ax.text(0.01, 0.035, footer, fontsize=11, color=GREY, style="italic", va="center")
    return fig, ax


# ---- one function per architecture ----------------------------------------------------------------
def draw_nn(model, cfg):
    fig, ax = setup("NN: plain neural network (baseline)", model,
                    "No encoders, no cross-frequency mixing: every frequency is predicted on its own. The input "
                    "order of the resonators matters (trained with a random order per batch).", mode_note=False)
    d = design_box(ax, rect(0, A), f"{NUM_RES} resonators\nflattened\n$[m_1, k_1, \\ldots, y_{NUM_RES}]$\n"
                   f"({5 * NUM_RES} numbers)")
    f = freq_box(ax)
    cat = box(ax, rect(1, (0.37, 0.22)), "Concatenate", f"$[$design$, f]$\n{5 * NUM_RES + 1} numbers\nfor every $f$",
              LIGHT["grey"])
    mlp = box(ax, rect(2, TALL, span=2), "MLP", f"{cfg['depth']} $\\times$ [Linear {cfg['hidden_dim']}, "
              f"{cfg['activation'].upper() if cfg['activation'] == 'relu' else cfg['activation']}, "
              f"Dropout {cfg['dropout']:g}]\n\nsame weights at every frequency\n\n{count(model.mlp):,} par.",
              LIGHT["purple"], PURPLE)
    out = output_box(ax, f"Linear {cfg['hidden_dim']} $\\to$ 1", model.output, rect(4, (0.35, 0.3), span=2))
    for a, b in ((d, cat), (f, cat), (cat, mlp), (mlp, out)):
        link(ax, a, b)
    return fig


def draw_don(model, cfg):
    P, T = cfg["basis_dim"], cfg["num_terms"]
    fig, ax = setup("DON: DeepONet", model,
                    "Branch $\\times$ trunk: the ERP is a sum of design-dependent coefficients times basis functions "
                    "of $f$; FiLM lets the basis itself change with the design, so peaks can move.")
    d, f = design_box(ax), freq_box(ax)
    s = set_encoder(ax, model.configuration_encoder, cfg["context_dim"])
    br = box(ax, rect(2, A), "Branch MLP", f"$c \\to b_{{j,p}}$\n{T} terms $\\times$ {P}\ncoefficients\n"
             f"{count(model.branch_head):,} par.", LIGHT["blue"], BLUE)
    tr = box(ax, rect(1, C), "Trunk MLP", f"$f \\to \\varphi_{{j,p}}(f)$\n{T} $\\times$ {P} basis functions\n"
             f"{count(model.trunk):,} par.", LIGHT["green"], GREEN)
    film = box(ax, rect(2, C), "FiLM", "$\\gamma\\,\\varphi + \\beta$\n$\\gamma = 1 + 0.25\\tanh(\\cdot)$\n"
               f"$\\gamma, \\beta$ from $c$\n{count(model.trunk_modulation):,} par.", LIGHT["green"], GREEN)
    core = box(ax, rect(3, TALL), "Term sum", f"per term $j$:\n$\\sum_p b_{{j,p}}\\,\\varphi_{{j,p}}(f)/\\sqrt{{{P}}}$"
               f"\n\nsoftmax weights $w_j$\n$\\sum_j w_j(\\cdot)$ + bias", LIGHT["purple"], PURPLE)
    ref = refinement(ax, model.frequency_refinement, cfg["refine_width"])
    out = output_box(ax, "Linear (residual)", None)
    for a, b in ((d, s), (s, br), (f, tr), (tr, film), (br, core), (film, core), (core, ref), (ref, out)):
        link(ax, a, b)
    _bank.arrow(ax, (s[0] + s[2] / 2, s[1]), (film[0] + 0.02, film[1] + film[3]), color=BLUE, ls="--")
    ax.text(0.33, 0.5, "$\\gamma, \\beta$", color=BLUE, fontsize=11)
    return fig


def draw_dno(model, cfg):
    fig, ax = setup("DNO: deep neural operator", model,
                    "A deep residual network evaluated at every frequency; the design and the detuning features "
                    "modulate every block (FiLM), and a local convolution mixes neighbouring frequencies.")
    d, f = design_box(ax), freq_box(ax)
    s = set_encoder(ax, model.configuration_encoder, cfg["context_dim"])
    q = query_encoder(ax, model.resonance_query, cfg["query_dim"])
    fe = box(ax, rect(1, C), "Frequency encoder", f"MLP 1 $\\to$ {cfg['frequency_dim']} $\\to$ "
             f"{cfg['frequency_dim']}\n{count(model.frequency_encoder):,} par.", LIGHT["green"], GREEN)
    lift_in = cfg["context_dim"] + cfg["frequency_dim"] + cfg["query_dim"]
    lift = box(ax, rect(2, TALL), "Lift", f"concatenate\n$[c, $ emb$(f), q(f)]$\n({lift_in})\n\nLinear $\\to$ "
               f"{cfg['hidden_dim']}\nSiLU\n\n{count(model.lift):,} par.", LIGHT["purple"], PURPLE)
    core = box(ax, rect(3, TALL), f"{cfg['depth']} $\\times$ FiLM residual block",
               "$h \\leftarrow \\sigma(\\gamma \\odot \\mathrm{ResMLP}(h) + \\beta)$\n\n$\\gamma = 1 + 0.2\\tanh(\\cdot)$\n"
               "$\\beta = 0.1\\,(\\cdot)$\n$\\gamma, \\beta$ from $[c, q(f)]$\n\npointwise in $f$\n"
               f"width {cfg['hidden_dim']}\n{count(model.blocks):,} par.", LIGHT["purple"], PURPLE)
    ref = refinement(ax, model.frequency_refinement, cfg["hidden_dim"])
    h = cfg["hidden_dim"]
    out = output_box(ax, f"MLP {h} $\\to$ {h // 2} $\\to$ 1", model.output)
    for a, b in ((d, s), (d, q), (f, q), (f, fe), (s, lift), (q, lift), (fe, lift), (lift, core), (core, ref),
                 (ref, out)):
        link(ax, a, b)
    condition(ax, s, core, "condition $[c, q(f)]$")
    return fig


def _lift_core_out(ax, model, cfg, width, lift_in, core_title, core_body, core_module, head_dims, head_module,
                   refine_module=None):
    """Shared right half: lift (col 2), core (col 3, or cols 3-4 without refinement), [refinement], output."""
    lift = box(ax, rect(2, TALL), "Lift", f"concatenate\n$[c, q(f), f]$\n({lift_in})\n\nLinear $\\to$ {width}\n\n"
               f"{count(model.lift):,} par.", LIGHT["purple"], PURPLE)
    if refine_module is None:
        core = box(ax, rect(3, TALL, span=2), core_title, core_body + f"\n{count(core_module):,} par.",
                   LIGHT["purple"], PURPLE)
        out = output_box(ax, head_dims, head_module)
        link(ax, core, out)
    else:
        core = box(ax, rect(3, TALL), core_title, core_body + f"\n{count(core_module):,} par.", LIGHT["purple"],
                   PURPLE)
        ref = refinement(ax, refine_module, width)
        out = output_box(ax, head_dims, head_module)
        link(ax, core, ref)
        link(ax, ref, out)
    link(ax, lift, core)
    return lift


def draw_fno(model, cfg):
    w = cfg["width"]
    fig, ax = setup("FNO: Fourier neural operator", model,
                    "Global mixing along the frequency axis in Fourier space (the lowest modes), plus a local "
                    "convolution for sharp peaks; the axis is padded so the FFT does not wrap the band edges.")
    d, f = design_box(ax), freq_box(ax)
    s = set_encoder(ax, model.configuration_encoder, w)
    q = query_encoder(ax, model.resonance_query, cfg["query_dim"])
    lift = _lift_core_out(ax, model, cfg, w, w + cfg["query_dim"] + 1,
                          f"{cfg['depth']} $\\times$ Fourier block",
                          f"pad the frequency axis by {cfg['padding']} (replicate)\n\n"
                          "$x \\leftarrow \\sigma(\\mathrm{GN}(x + K x + W x))$\n\n"
                          f"$K$: FFT, keep the lowest {cfg['modes']} modes,\nlearned complex weights, inverse FFT\n"
                          "$W$: Conv1d ($k = 3$), local\nGroupNorm, GELU, dropout "
                          f"{cfg['dropout']:g}\n\ncrop the padding\nwidth {w}", model.blocks,
                          f"MLP {w} $\\to$ {w} $\\to$ 1", model.project)
    for a, b in ((d, s), (d, q), (f, q), (s, lift), (q, lift), (f, lift)):
        link(ax, a, b)
    return fig


def draw_dco(model, cfg):
    h = cfg["hidden_dim"]
    fig, ax = setup("DCO: deep cat operator", model,
                    "Branch (design), trunk (frequency) and resonance-query features are concatenated (\"cat\") "
                    "and passed through a deep residual MLP at every frequency, then refined locally.")
    d, f = design_box(ax), freq_box(ax)
    bd = cfg["branch_dim"]
    if model.use_sorted_branch:  # DCO keeps both branches separate and concatenates them in the lift
        s = box(ax, rect(1, A), "Branch: set + sorted", "per resonator: $[m, k, f_t, x, y]$\n+ 120 plate mode shapes\n"
                f"set: shared MLP, mean + max ({bd})\nsorted: order by $f_t$, concat.,\nMLP ({bd})\n"
                f"{count(model.branch, model.sorted_branch):,} par.", LIGHT["blue"], BLUE)
    else:
        s = set_encoder(ax, model.branch, bd, title="Branch: set encoder")
    q = query_encoder(ax, model.resonance_query, cfg["query_dim"])
    tr = box(ax, rect(1, C), "Trunk MLP", f"$f$: MLP 1 $\\to$ {h} $\\to$ {cfg['trunk_dim']}\n"
             f"{count(model.trunk):,} par.", LIGHT["green"], GREEN)
    lift_in = bd * (2 if model.use_sorted_branch else 1) + cfg["trunk_dim"] + cfg["query_dim"]
    branches = "$c_{set}, c_{sort}$" if model.use_sorted_branch else "$c$"
    lift = box(ax, rect(2, TALL), "Lift", f"concatenate\n$[${branches}$,$\ntrunk$(f), q(f)]$\n({lift_in})\n\nLinear $\\to$ {h}"
               f"\nSiLU\n\n{count(model.lift):,} par.", LIGHT["purple"], PURPLE)
    core = box(ax, rect(3, TALL), f"{cfg['depth']} $\\times$ residual MLP block",
               "$h \\leftarrow \\sigma(\\mathrm{LN}(h + \\mathrm{MLP}(h)))$\n\npointwise in $f$\n"
               f"width {h}\n{count(model.blocks):,} par.", LIGHT["purple"], PURPLE)
    ref = refinement(ax, model.frequency_refinement, h)
    out = output_box(ax, f"MLP {h} $\\to$ {h // 2} $\\to$ 1", model.output)
    for a, b in ((d, s), (d, q), (f, q), (f, tr), (s, lift), (q, lift), (tr, lift), (lift, core), (core, ref),
                 (ref, out)):
        link(ax, a, b)
    return fig


def draw_gno(model, cfg):
    w, fd = cfg["width"], cfg["frequency_dim"]
    nh = cfg["modal_harmonics"]
    feat = 5 + 2 * nh + nh * nh
    fig, ax = setup("GNO: graph neural operator", model,
                    "The resonators are the nodes of a complete graph; message passing lets them interact, and every "
                    "frequency then attends to the resonators (a learned kernel integral over the graph).")
    d, f = design_box(ax), freq_box(ax)
    nl = box(ax, rect(1, A), "Node lift", f"per resonator:\n$[m, k, f_t, x, y]$\n+ {feat - 5} plate mode shapes\n({feat})\n"
             f"MLP $\\to$ {w}\n{count(model.node_lift):,} par.", LIGHT["blue"], BLUE)
    gl = box(ax, rect(2, (0.50, 0.36)), f"{cfg['depth']} $\\times$ message passing", "complete graph\nof resonators\n\n"
             "edges: $\\Delta[m, k, f_t, x, y]$,\ndistance, $|\\Delta f_t|$\n\nmessage MLP, mean,\nupdate MLP,\n"
             f"LayerNorm residual\n$\\to$ nodes $h_i$ ({w})\n{count(model.layers):,} par.", LIGHT["blue"], BLUE)
    fe = box(ax, rect(1, C), "Frequency encoder", f"MLP 1 $\\to$ {fd} $\\to$ {fd}\n"
             f"{count(model.frequency_encoder):,} par.", LIGHT["green"], GREEN)
    core = box(ax, rect(3, TALL), "Kernel + attention", "for every $f$ and resonator $i$:\n"
               "$\\kappa_i(f) = $ MLP$[h_i$, raw$_i$,\nemb$(f), f, \\delta, |\\delta|, \\delta^2]$\n\n"
               "$\\alpha_i(f) = $ softmax$_i$ MLP$(\\kappa_i)$\n\nintegral over the graph:\n"
               f"$\\sum_i \\alpha_i(f)\\,\\kappa_i(f)$ ({w})\n\n"
               f"{count(model.query_kernel) + count(model.attention_score):,} par.", LIGHT["purple"], PURPLE)
    ref = refinement(ax, model.frequency_refinement, w)
    out = output_box(ax, f"MLP {w} $\\to$ {w // 2} $\\to$ 1", model.output)
    for a, b in ((d, nl), (f, fe), (nl, gl), (gl, core), (fe, core), (core, ref), (ref, out)):
        link(ax, a, b)
    ax.plot([d[0] + d[2], 0.33], [d[1] + 0.03, 0.45], color=GREY, lw=1.4, ls=":", zorder=4)
    _bank.arrow(ax, (0.33, 0.45), (core[0], 0.45), color=GREY, ls=":")
    ax.text(0.42, 0.455, "raw resonators ($f_t$ for $\\delta$)", fontsize=10.5, color=GREY, ha="center", va="bottom")
    return fig


def draw_sto(model, cfg):
    w, nh = cfg["width"], cfg["modal_harmonics"]
    feat = 5 + 2 * nh + nh * nh
    fig, ax = setup("STO: set transformer operator", model,
                    "Self-attention lets the resonators interact; every frequency then queries the resonators by "
                    "cross-attention, biased by how far $f$ is from each tuning frequency.")
    d, f = design_box(ax), freq_box(ax)
    nl = box(ax, rect(1, A), "Node lift", f"per resonator:\n$[m, k, f_t, x, y]$\n+ {feat - 5} plate mode shapes\n({feat})\n"
             f"MLP $\\to$ {w}\n{count(model.node_lift):,} par.", LIGHT["blue"], BLUE)
    enc = box(ax, rect(2, (0.37, 0.49)), "Transformer encoder", f"{len(model.encoder.layers)} layers, {cfg['heads']} "
              f"heads\n\nself-attention\nbetween resonators\n\nfeed-forward {cfg['ff_dim']}\ndropout {cfg['dropout']:g}\n"
              f"$\\to$ resonator tokens\n{count(model.encoder):,} par.", LIGHT["blue"], BLUE)
    fq = box(ax, rect(1, C), "Frequency query", f"MLP 1 $\\to$ {w} $\\to$ {w}\n{count(model.frequency_query):,} par.",
             LIGHT["green"], GREEN)
    core = box(ax, rect(3, TALL), "Cross-attention", f"queries: frequencies\nkeys, values: tokens\n{cfg['heads']} heads\n\n"
               "+ detuning bias per head:\nMLP$[m, k, f_t, x, y, f, \\delta, |\\delta|]$\n\nsoftmax over resonators\n"
               f"residual + LayerNorm\n{count(model.cross_attention) + count(model.cross_norm):,} par.",
               LIGHT["purple"], PURPLE)
    mix = box(ax, rect(4, (0.35, 0.3)), "Frequency mixer", "Conv1d ($k = 5$)\n+ dilated Conv1d\n($k = 3$, $d = 2$)\n"
              f"residual, GN, GELU\n{count(model.frequency_mixer):,} par.", LIGHT["purple"], PURPLE)
    out = output_box(ax, f"MLP {w} $\\to$ {w}\n$\\to$ {w // 2} $\\to$ 1", model.output)
    for a, b in ((d, nl), (f, fq), (nl, enc), (enc, core), (fq, core), (core, mix), (mix, out)):
        link(ax, a, b)
    return fig


def draw_siren(model, cfg):
    hd = cfg["hidden_dim"]
    fig, ax = setup("SIREN: sinusoidal representation network operator", model,
                    "A continuous function of frequency built from sine layers (good at sharp, oscillatory "
                    "features); the design modulates every layer.")
    d, f = design_box(ax), freq_box(ax)
    s = set_encoder(ax, model.configuration_encoder, cfg["context_dim"])
    q = query_encoder(ax, model.resonance_query, cfg["query_dim"])
    inp = box(ax, rect(2, (0.24, 0.35)), "Input", f"$[f, q(f)]$\n({1 + cfg['query_dim']} numbers)", LIGHT["grey"])
    core = box(ax, rect(3, TALL), f"{cfg['depth']} $\\times$ modulated sine layer",
               "$h \\leftarrow \\sin\\big(\\omega_0\\,(\\gamma \\odot (W h + b) + \\beta)\\big)$\n\n"
               f"$\\omega_0 = {cfg['omega_0']:g}$, width {hd}\n\n$\\gamma = 1 + 0.25\\tanh(\\cdot)$\n"
               "$\\beta = 0.1\\,(\\cdot)$\n$\\gamma, \\beta$ from context $c$\n\npointwise in $f$\n"
               f"{count(model.layers):,} par.", LIGHT["purple"], PURPLE)
    ref = refinement(ax, model.frequency_refinement, hd)
    out = output_box(ax, f"Linear {hd} $\\to$ 1", model.output)
    for a, b in ((d, s), (d, q), (f, q), (q, inp), (f, inp), (inp, core), (core, ref), (ref, out)):
        link(ax, a, b)
    condition(ax, s, core, "modulation from $c$")
    return fig


def draw_wno(model, cfg):
    w = cfg["width"]
    fig, ax = setup("WNO: wavelet neural operator", model,
                    "Mixing along the frequency axis on several scales at once (Haar wavelets): coarse levels see the "
                    "whole band, fine levels keep sharp, local peaks.")
    d, f = design_box(ax), freq_box(ax)
    s = set_encoder(ax, model.configuration_encoder, w)
    q = query_encoder(ax, model.resonance_query, cfg["query_dim"])
    lift = _lift_core_out(ax, model, cfg, w, w + cfg["query_dim"] + 1, f"{cfg['depth']} $\\times$ wavelet block",
                          f"{cfg['levels']}-level Haar transform:\nlow, high $= (x_{{2i}} \\pm x_{{2i+1}})/\\sqrt{{2}}$\n\n"
                          "Conv1d ($k = 3$) on every band\n+ Conv1d on the coarsest level\n\ninverse Haar transform\n"
                          "$x \\leftarrow \\sigma(\\mathrm{GN}(x + $ wavelet$(x) + W x))$\n"
                          f"GELU, dropout {cfg['dropout']:g}\nwidth {w}", model.blocks,
                          f"MLP {w} $\\to$ {w} $\\to$ 1", model.project)
    for a, b in ((d, s), (d, q), (f, q), (s, lift), (q, lift), (f, lift)):
        link(ax, a, b)
    return fig


def draw_lno(model, cfg):
    w, P = cfg["width"], cfg["num_poles"]
    fig, ax = setup("LNO: Laplace neural operator", model,
                    "The design predicts a learned pole-residue (rational) basis in the Laplace domain, the natural "
                    "form of a resonant response; the poles are learned features, not the plate's physical poles.")
    d, f = design_box(ax), freq_box(ax)
    s = set_encoder(ax, model.configuration_encoder, w)
    q = query_encoder(ax, model.resonance_query, cfg["query_dim"])
    pp = box(ax, rect(2, (0.37, 0.49)), "Pole predictor", f"MLP {w} $\\to$ {cfg['pole_hidden']} $\\to$ "
             f"{cfg['pole_hidden']} $\\to$ {4 * P}\n\n{P} poles\n$p_k = -\\sigma_k + i\\,\\omega_k$\n"
             "$\\sigma_k, \\omega_k > 0$ (softplus)\n\nresidues $r_k \\in \\mathbb{C}$\n"
             f"{count(model.pole_predictor):,} par.", LIGHT["blue"], BLUE)
    core = box(ax, rect(3, TALL), "Pole features + lift", "$s = i f$\n\n"
               "$\\frac{r_k}{s - p_k} + \\frac{\\bar r_k}{s - \\bar p_k}$\n\n"
               f"real and imaginary parts\n$\\to$ {2 * P} features\n\nconcatenate $q(f)$, $f$\n"
               f"Linear $\\to$ {w}, dropout {cfg['dropout']:g}\n{count(model.lift):,} par.", LIGHT["purple"], PURPLE)
    ref = refinement(ax, model.refine, w)
    out = output_box(ax, f"MLP {w} $\\to$ {w} $\\to$ 1", model.project)
    for a, b in ((d, s), (d, q), (f, q), (s, pp), (pp, core), (q, core), (f, core), (core, ref), (ref, out)):
        link(ax, a, b)
    return fig


DRAW = {"NN": draw_nn, "DON": draw_don, "DNO": draw_dno, "FNO": draw_fno, "DCO": draw_dco, "GNO": draw_gno,
        "STO": draw_sto, "SIREN": draw_siren, "WNO": draw_wno, "LNO": draw_lno}


def main(names) -> None:
    specs = {spec["short"]: spec for spec in OPERATORS.values()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = specs[name]
        # the final trained models: set + f_t-sorted encoder, physical plate-mode
        # features (NN: random resonator order); same weights layout as their checkpoints
        _, overrides = forward_variant(spec, FINAL_VARIANT)
        cfg = {**spec["model_config"], **overrides}
        model = build_operator_model(spec["build_model"], NUM_RES, cfg)
        fig = DRAW[name](model, cfg)
        path = OUT_DIR / f"{name.lower()}_architecture.png"
        save_figure(fig, path)
        plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or list(DRAW))
