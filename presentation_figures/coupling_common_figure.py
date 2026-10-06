"""Common form of the eight invertible coupling-flow operators (iFNO, iDCO, iGNO, iDNO, iWNO, iLNO, iSIREN, iSTO).

Two blocks change with the architecture: the lift P (where the design enters) and the gate network L of the coupling
blocks.  Parameter counts: each operator built with erp_invertible.scripts.train.RECOMMENDED_OPTIONS (2 resonators,
design_dim 8).  Loss terms and stages: erp_invertible/scripts/train.py and common.py.
"""
from common_forms_kit import *  # noqa: F401,F403
from common_forms_kit import elem, card, frame, loss_panel, FWD_COL, INV_COL, VARY_FILL, VARY_EDGE

fig, ax = new_figure()
rx = frame(ax, 71.2, "Invertible coupling-flow operators: common structure",
           r"one coupling stack run forwards (design $\rightarrow$ ERP) and backwards (ERP $\rightarrow$ designs); the lift $P$ and the gate $L$ change")

FY0, FY1 = 40.2, 48.4
IY0, IY1 = 30.0, 38.2
FM, IM = (FY0 + FY1) / 2, (IY0 + IY1) / 2
# ---- forward row (blue, solid) ------------------------------------------------------------------
elem(ax, 2.3, 8.7, FY0, FY1, r"Design $\mathbf{a}$", "$D=8$", BLUE, size=7.4, tsize=8.4)
elem(ax, 10.1, 22.0, FY0, FY1, "Lift $P$", "design encoder, query,\nfrequency $\\rightarrow$ latent $v_0$\n(changes with the model)", BLUE, size=6.8, tsize=8.6, vary=True)
elem(ax, 23.4, 31.0, FY0, FY1, "Latent", "$(v_1,v_2)$\n$2\\times24$ ch.\n$\\times$ 301 $f$", BLUE, size=6.8, tsize=8.4)
elem(ax, 51.8, 60.6, FY0, FY1, "Readout $Q$", "per frequency\nMLP $48\\rightarrow48\\rightarrow1$", BLUE, size=6.8, tsize=8.4)
panel(ax, 62.0, FY0, 8.4, FY1 - FY0, fc=ORANGE, r=1.3, z=2)
text(ax, 66.2, FY1 - 1.2, "ERP", size=8.4)
spectrum_inset(fig, ax, 63.0, FY0 + 0.9, 6.4, 3.9, sample_erp())
for xa, xb in ((8.7, 10.1), (22.0, 23.4), (31.0, 32.4), (50.4, 51.8), (60.6, 62.0)):
    arrow(ax, [(xa, FM), (xb, FM)], color=FWD_COL, lw=1.3, ms=8)
# ---- inverse row (orange, dashed) ---------------------------------------------------------------
elem(ax, 62.0, 70.4, IY0, IY1, "Target $y$", "ERP (301)", YELLOW, size=7.2, tsize=8.4)
elem(ax, 51.8, 60.6, IY0, IY1, "Lift $P'$", "per frequency\n$[y(f),\\mathrm{MLP}(f)]$\n$\\rightarrow$ 48 channels", ORANGE, size=6.8, tsize=8.4)
elem(ax, 22.8, 31.0, IY0, IY1, "Readout $Q'$", "pool over $f$:\nmean, max, bins\n$\\rightarrow$ MLP $\\rightarrow\\hat{\\mathbf{a}}$", ORANGE, size=6.5, tsize=8.4)
elem(ax, 12.0, 21.2, IY0, IY1, r"$\beta$-VAE", "$\\hat{\\mathbf{a}}\\rightarrow\\mu,\\sigma$ ($z$: 6)\nsample, decode", ORANGE, size=6.6, tsize=8.4)
elem(ax, 2.3, 10.6, IY0, IY1, "Designs", "several\nsamples $\\hat{\\mathbf{a}}$", ORANGE, size=6.8, tsize=8.4)
for xa, xb in ((62.0, 60.6), (51.8, 50.4), (32.4, 31.0), (22.8, 21.2), (12.0, 10.6)):
    arrow(ax, [(xa, IM), (xb, IM)], color=INV_COL, lw=1.3, ls="--", ms=8)
# ---- coupling stack with the gate slot ----------------------------------------------------------
panel(ax, 32.4, IY0, 18.0, FY1 - IY0, fc=PURPLE, r=1.6, z=1)
text(ax, 41.4, 47.2, "Coupling stack", size=8.8)
text(ax, 41.4, 45.8, "4 blocks, same weights both ways", size=6.3, color="#444444")
text(ax, 41.4, 44.3, r"$v_1'=v_1\odot S(L(v_2))$", size=6.9, color=FWD_COL)
text(ax, 41.4, 42.9, r"$v_2'=v_2\odot S(L(v_1'))$", size=6.9, color=FWD_COL)
text(ax, 41.4, 41.5, r"$v_2=v_2'\,/\,S(L(v_1'))$", size=6.9, color=INV_COL)
text(ax, 41.4, 40.1, r"$v_1=v_1'\,/\,S(L(v_2))$", size=6.9, color=INV_COL)
panel(ax, 33.4, 30.8, 16.0, 7.6, fc=VARY_FILL, ec=VARY_EDGE, ls="--", lw=1.6, r=1.0, z=3)
text(ax, 41.4, 37.0, "Gate network $L$", size=8.2)
text(ax, 41.4, 33.9, "(changes with the model;\ndoes not see the design)\n$S(x)=e^{2\\tanh(ax)}$", size=6.4, linespacing=1.3)
text(ax, 36.1, 28.3, "Only the coupling stack is an exact bijection; lifts and readouts are learned maps tied together by the losses.", size=7.0, color="#333333")
text(ax, 36.1, 26.8, "Several designs per target; one is picked with the model's own forward direction (best predicted spectrum).", size=7.0, color="#333333")
text(ax, 36.1, 24.7, "Lift $P$ and gate $L$ of each operator", size=9.4)

CARDS = [
    ("iFNO: Fourier", ["P: sorted encoder, query,", "Linear $[c,q(f),f]\\rightarrow48$", "L: Fourier layer: spectral conv", "(48 modes) + Conv1d, GELU"], "188,405 parameters"),
    ("iDCO: Deep Cat", ["P: branch encoder, trunk MLP($f$),", "query $\\rightarrow$ Linear", "L: pointwise residual MLP block", "+ local refinement"], "81,909 parameters"),
    ("iGNO: Graph", ["P: graph nodes + message passing,", "kernel and attention per $f$", "L: pointwise MLP", "+ local refinement"], "62,210 parameters"),
    ("iDNO: Deep Neural Op.", ["P: encoder, frequency encoder,", "resonance query $\\rightarrow$ Linear", "L: FiLM residual block (FiLM from", "a frequency embedding) + refinement"], "84,389 parameters"),
    ("iWNO: Wavelet", ["P: encoder, resonance query,", "$f\\rightarrow$ Linear", "L: multi-level Haar wavelet", "block"], "115,213 parameters"),
    ("iLNO: Laplace", ["P: encoder $\\rightarrow$ design-dependent", "poles, pole features, query", "L: learned design-independent poles,", "pointwise transform, refinement"], "83,229 parameters"),
    ("iSIREN: SIREN", ["P: $[f,q(f)]\\rightarrow$ sine layers modulated", "by the context $\\rightarrow$ Linear", "L: sine layer", "+ local refinement"], "90,645 parameters"),
    ("iSTO: Set Transformer", ["P: resonator tokens, detuning-biased", "cross-attention $\\rightarrow$ Linear", "L: frequency mixer (Conv1d, k 5,", "plus dilated Conv1d)"], "71,089 parameters"),
]
CW, GAP = 16.4, 0.7
for i, (t, lines, par) in enumerate(CARDS):
    r, c = divmod(i, 4)
    card(ax, 2.3 + c * (CW + GAP), 14.2 if r == 0 else 5.0, CW, 8.8, t, lines, par, size=7.2, tsize=7.9)

loss_panel(ax, rx, [
    ("Forward loss $J_{FWD}$", r"MSE$(\hat y,y)$, $\hat y=Q(\mathrm{stack}(P(\mathbf{a})))$"),
    ("Inverse loss $J_{INV}$", r"MSE$(\hat{\mathbf{a}},\mathbf{a})$, $\hat{\mathbf{a}}$ via $P'$, stack$^{-1}$, $Q'$"),
    ("Round trips $J_{PQ'}$, $J_{P'Q}$", "lift and readout only (stack skipped)"),
    (r"Cycle $0.1\,J_{cyc}$, alignment $0.1\,J_{align}$", r"$\mathbf{a}\rightarrow\hat y\rightarrow\mathbf{a}$;  $\Vert u_0-v_K\Vert^2/\Vert v_K\Vert^2$"),
    (r"$\beta$-VAE loss", r"MSE(recon, $\mathbf{a}$) $+\,\beta\,\mathrm{KL}$, $\beta=0.05$"),
], "3 stages (Adam, batch 128, cosine):\n1: all but the VAE, 20 epochs,\nlr $5\\times10^{-4}$\n2: VAE alone, 25 epochs, lr $10^{-3}$\n3: everything, 15 epochs,\nlr $3\\times10^{-4}$", data_label="Training data: pairs ($\\mathbf{a}$, $y$)", box_h=4.9, footer_size=6.9)
text(ax, 50.0, 2.9, "Recommended configuration (2 resonators): bounded design coordinates, bounded gate, binned readout, padded FFT (iFNO), cycle and alignment terms, VAE on the stage-1 estimates; "
                    "$f_t$-sorted encoder and plate-mode features where the model has an encoder.", size=7.2, color="#444444")
text(ax, 50.0, 1.3, "Parameter counts: each operator built with these options.  Data: 2 resonators, 18 plate modes.", size=7.2, color="#444444")
save(fig, "invertible_coupling_common")
