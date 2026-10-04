"""Common form of all ten forward neural operators; the operator core is the general, changing block.

Facts: model_config / parameter counts of the ten checkpoints in models/GENERAL/100k (*_sorted_phys, *_phys, nn_perm);
structures from erp_forward_operators/*.py (forward() of each class).  Training: AdamW 5e-4 (SIREN 2e-4), cosine,
200 epochs, batch 128, loss MSE + 0.5 slope + 0.05 peak, 100k configurations, 80/10/10.
"""
from common_forms_kit import *  # noqa: F401,F403
from common_forms_kit import elem, card, frame, loss_panel

fig, ax = new_figure()
rx = frame(ax, 71.2, "Forward neural operators: common structure",
           r"resonator design and frequency $\rightarrow$ ERP spectrum; only the operator core changes between architectures")

# ---- common pipeline ---------------------------------------------------------------------------
Y0, Y1 = 33.2, 48.3
elem(ax, 2.3, 9.3, Y0, Y1, "Inputs", "resonators $\\mathbf{r}_i$ (3):\n$[m,k,f_t,x,y]$\nz-scored\n\nfrequency $f$\n301 points", GRAY, size=7.8, tsize=9.2)
elem(ax, 10.9, 21.3, Y0, Y1, "Design encoder", "permutation-\ninvariant encoding\nof the resonator\nset $\\rightarrow$ context $c$", BLUE, size=7.8, tsize=9.2)
elem(ax, 22.9, 33.3, Y0, Y1, "Resonance query", "per resonator and $f$:\n$\\delta=f-f_t$, $|\\delta|$,\n$\\delta^2$; pooled over\nresonators $\\rightarrow q(f)$", GREEN, size=7.8, tsize=9.2)
elem(ax, 34.9, 49.5, Y0, Y1, "Operator core", "combines the design\nfeatures with the\nfrequency\n\n(changes with the\narchitecture)", ORANGE, size=7.8, vary=True, tsize=9.2)
elem(ax, 51.1, 60.5, Y0, Y1, "Local refinement", "mixes neighbouring\nfrequencies along $f$\n(Conv1d,\nGroupNorm)", PURPLE, size=7.8, tsize=9.2)
elem(ax, 62.1, 68.5, Y0, Y1, "Output\nhead", r"$\rightarrow$ ERP" + "\n" + r"$\hat y(f)$", GRAY, size=7.8, tsize=9.2)
M = (Y0 + Y1) / 2
for xa, xb in ((9.3, 10.9), (21.3, 22.9), (33.3, 34.9), (49.5, 51.1), (60.5, 62.1)):
    arrow(ax, [(xa, M), (xb, M)], lw=1.2, ms=8)
text(ax, 35.7, 31.6, r"Which optional blocks (query, refinement) a model uses is listed on its card.  The frequency $f$ enters the query and the core.",
     size=7.2, color="#444444")
text(ax, 36.1, 29.9, "Operator core of each architecture", size=9.6)

# ---- variant cards ----------------------------------------------------------------------------
CARDS = [
    ("NN: plain network", ["encoder: none; flattened", "$\\mathbf{r}_i$ and $f$ enter one MLP", "core: 6 dense layers (148),", "ReLU, dropout 0.1", "query: no; refinement: no"], "112,925 parameters"),
    ("DON: DeepONet", ["encoder: set + sorted (71)", "core: branch coefficients (4 terms", "$\\times$ 113) $\\cdot$ trunk MLP($f$);", "trunk FiLM-modulated by $c$", "query: no; refinement: yes"], "146,294 parameters"),
    ("DNO: Deep Neural Op.", ["encoder: set + sorted (57)", "core: 4 FiLM residual blocks", "(width 54), FiLM from the", "context and query", "query: yes; refinement: yes"], "149,994 parameters"),
    ("FNO: Fourier NO", ["encoder: set + sorted (23)", "core: 4 Fourier blocks, 35", "modes (spectral + local conv),", "frequency axis padded by 8", "query: yes; refinement: in blocks"], "153,564 parameters"),
    ("DCO: Deep Cat Operator", ["encoder: set + sorted (56+56)", "core: concatenate branch, trunk", "MLP($f$) and query (196) $\\rightarrow$", "lift $\\rightarrow$ 4 residual blocks (67)", "query: yes; refinement: yes"], "155,240 parameters"),
    ("GNO: Graph NO", ["encoder: resonators as graph", "nodes, 3 message-passing layers", "core: kernel MLP on node, $f$,", "$\\delta$; attention over resonators", "refinement: yes"], "112,230 parameters"),
    ("STO: Set Transformer Op.", ["encoder: resonator tokens,", "2 Transformer layers (4 heads)", "core: cross-attention from", "frequencies to tokens (detuning", "bias); own frequency mixer"], "110,193 parameters"),
    ("SIREN: SIREN NO", ["encoder: set + sorted (57)", "core: 4 sine layers ($\\omega_0=20$)", "on $[f,q(f)]$, FiLM-modulated", "by the context $c$", "query: yes; refinement: yes"], "152,746 parameters"),
    ("WNO: Wavelet NO", ["encoder: set + sorted (57)", "core: 4 wavelet blocks", "(3-level Haar transform, learned", "mixing per level)", "query: yes; refinement: in blocks"], "145,946 parameters"),
    ("LNO: Laplace NO", ["encoder: set + sorted (90)", "core: 13 poles and residues", "predicted from $c$; features", "$r/(s-p)+\\bar r/(s-\\bar p)$, $s=jf$", "query: yes; refinement: yes"], "162,704 parameters"),
]
CW, CH, GAP = 13.0, 11.6, 0.7
for i, (t, lines, par) in enumerate(CARDS):
    r, c = divmod(i, 5)
    card(ax, 2.3 + c * (CW + GAP), (17.2 if r == 0 else 5.1), CW, CH, t, lines, par)

# ---- shared loss -------------------------------------------------------------------------------
loss_panel(ax, rx, [
    ("MSE loss", r"$\frac{1}{F}\sum_f(\hat y_f-y_f)^2$"),
    (r"Slope loss ($\times\,0.5$)", r"$\frac{1}{F-1}\sum_f(\Delta\hat y_f-\Delta y_f)^2$"),
    (r"Peak loss ($\times\,0.05$)", r"$\sum_{f\in\mathcal{P}(y)}(\hat y_f-y_f)^2$"),
], "Same for all ten:\nAdamW, cosine decay, 200 epochs,\nbatch 128, lr $5\\times10^{-4}$\n(SIREN: $2\\times10^{-4}$)\n100k configurations, 80/10/10\n$\\mathcal{P}(y)$: true resonance peaks", data_label="Training data: true ERP $y(f)$")

text(ax, 50.0, 2.9, "Set encoder: shared MLP, mean + max pooling; sorted encoder: resonators ordered by $f_t$, concatenated, MLP; each resonator carries 120 plate-mode features.  "
                    "Numbers in brackets are widths.", size=7.6, color="#444444")
text(ax, 50.0, 1.3, "GNO and STO have no set / sorted encoder: their resonators are graph nodes or attention tokens.  ERP values are normalised before the loss.", size=7.6, color="#444444")
save(fig, "forward_operators_common")
