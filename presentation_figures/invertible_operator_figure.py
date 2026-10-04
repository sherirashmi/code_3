"""Presentation figure: invertible Fourier neural operator (iFNO, recommended configuration) and its losses.

Sizes are counted on IFNO(design_dim=8, num_res=2) built with erp_invertible_operators.train.RECOMMENDED_OPTIONS:
lift P 43,792 (sorted encoder 30,448 + resonance query 9,408 + Linear 3,936), lift P' 2,752, coupling stack 117,796,
readout Q 2,401, readout Q' 11,020, beta-VAE 10,644; total 188,405.  Training: 3 stages (20 / 25 / 15 epochs),
Adam, batch 128 (committed iFNO checkpoint's training_config).
"""
import numpy as np

from diagram_kit import *  # noqa: F401,F403
from diagram_kit import BLUE, GREEN, ORANGE, YELLOW, GRAY, PURPLE, PINK, OUT

fig, ax = new_figure()
panel(ax, 1.0, 4.4, 71.2, 49.6, fc="#f6f6f6", ls="--", lw=1.2, r=2.0, z=0)
panel(ax, 73.4, 4.4, 25.8, 49.6, fc="#f6f6f6", ls="--", lw=1.2, r=2.0, z=0)
text(ax, 36.6, 52.4, "Invertible operator: iFNO", size=12.5)
text(ax, 36.6, 50.6, r"one coupling stack run forwards (design $\rightarrow$ ERP) and backwards (ERP $\rightarrow$ designs);  188,405 parameters", size=8.3, color="#555555")
text(ax, 86.3, 52.4, "Loss functions", size=12.5)

FWD_COL, INV_COL = "#2a78d6", "#e8632b"
FY0, FY1 = 34.0, 47.5
IY0, IY1 = 7.5, 21.0
FM, IM = (FY0 + FY1) / 2, (IY0 + IY1) / 2


def elem(x0, x1, y0, y1, title, body, fc, size=6.9, tsize=8.6, params=None):
    panel(ax, x0, y0, x1 - x0, y1 - y0, fc=fc, r=1.3, z=2)
    text(ax, (x0 + x1) / 2, y1 - 1.3, title, size=tsize)
    text(ax, (x0 + x1) / 2, (y0 + y1) / 2 - 0.6, body, size=size, linespacing=1.3, color="#222222")
    if params:
        text(ax, (x0 + x1) / 2, y0 + 0.95, params, size=6.4, color="#555555")


# ---- forward row (blue, solid) ------------------------------------------------------------------
elem(2.3, 8.5, FY0, FY1, r"Design $\mathbf{a}$", "$D=8$:\n$[m,f_t,x,y]$\n$\\times$ 2 resonators", BLUE)
elem(10.1, 21.7, FY0, FY1, "Lift $P$", "sorted encoder $\\rightarrow c$ (48)\nresonance query $q(f)$ (32)\nLinear $[c,q(f),f]\\rightarrow48$", BLUE, params="43,792 parameters")
elem(23.3, 31.7, FY0, FY1, r"Latent $v_0$", "$(v_1,v_2)$\n$2\\times24$ channels\n$\\times$ 301 freq.", BLUE)
elem(50.3, 59.5, FY0, FY1, "Readout $Q$", "per frequency:\nMLP $48\\rightarrow48\\rightarrow1$", BLUE, params="2,401 par.")
panel(ax, 61.1, FY0, 9.4, FY1 - FY0, fc=ORANGE, r=1.3, z=2)
text(ax, 65.8, FY1 - 1.3, r"Predicted ERP", size=8.0)
spectrum_inset(fig, ax, 62.3, FY0 + 3.6, 7.0, 5.0, sample_erp())
text(ax, 65.8, FY0 + 1.6, r"$\hat y(f)$", size=8.2)
for xa, xb in ((8.5, 10.1), (21.7, 23.3), (31.7, 33.5), (48.5, 50.3), (59.5, 61.1)):
    arrow(ax, [(xa, FM), (xb, FM)], color=FWD_COL, lw=1.4)
text(ax, 49.4, FM + 1.3, r"$v_K$", size=7.6, color=FWD_COL)

# ---- inverse row (orange, dashed) ---------------------------------------------------------------
elem(2.3, 8.5, IY0, IY1, "Candidate\ndesigns", r"$\hat{\mathbf{a}}$ (8):" + "\nseveral\nsamples", ORANGE, tsize=7.6)
elem(10.0, 19.6, IY0, IY1, r"$\beta$-VAE", "$\\hat{\\mathbf{a}}_{\\rm point}\\rightarrow\\mu,\\sigma$ ($z$: 6)\nsample $z$,\ndecode $z\\rightarrow\\mathbf{a}$", ORANGE, params="10,644 par.", size=6.5)
elem(21.2, 31.7, IY0, IY1, "Readout $Q'$", "pool over $f$: mean, max,\n16 bins (160)\nMLP $160\\rightarrow64\\rightarrow8$", ORANGE, params="11,020 par.", size=6.7)
elem(50.3, 59.5, IY0, IY1, "Lift $P'$", "per frequency:\n$[y(f),\\mathrm{MLP}(f)]$ (33)\nLinear $\\rightarrow48$", ORANGE, params="2,752 par.", size=6.7)
panel(ax, 61.1, IY0, 9.4, IY1 - IY0, fc=YELLOW, r=1.3, z=2)
text(ax, 65.8, IY1 - 1.3, "Target ERP", size=8.0)
spectrum_inset(fig, ax, 62.3, IY0 + 3.6, 7.0, 5.0, sample_erp(137), color="black", fill=False)
text(ax, 65.8, IY0 + 1.6, r"$y(f)$", size=8.2)
for xa, xb in ((61.1, 59.5), (50.3, 48.5), (33.5, 31.7), (21.2, 19.6), (10.0, 8.5)):
    arrow(ax, [(xa, IM), (xb, IM)], color=INV_COL, lw=1.4, ls="--")
text(ax, 49.4, IM + 1.3, r"$u_0$", size=7.6, color=INV_COL)
text(ax, 32.6, IM + 1.3, r"$v_0$", size=7.6, color=INV_COL)

# ---- coupling stack ---------------------------------------------------------------------------
panel(ax, 33.5, 7.5, 15.0, 40.0, fc=PURPLE, r=1.6, z=1)
text(ax, 41.0, 45.7, "Coupling stack", size=9.4)
text(ax, 41.0, 44.1, "4 blocks, same weights both ways", size=6.6, color="#444444")
panel(ax, 34.4, 33.6, 13.2, 9.2, fc="white", r=0.9, z=3)
text(ax, 41.0, 41.6, "forward", size=6.8, color=FWD_COL)
text(ax, 41.0, 39.4, r"$v_1'=v_1\odot S(L(v_2))$", size=7.2)
text(ax, 41.0, 37.4, r"$v_2'=v_2\odot S(L(v_1'))$", size=7.2)
panel(ax, 34.4, 23.4, 13.2, 9.4, fc="white", r=0.9, z=3)
text(ax, 41.0, 31.2, "inverse (reverse order)", size=6.8, color=INV_COL)
text(ax, 41.0, 29.0, r"$v_2=v_2'\,/\,S(L(v_1'))$", size=7.2)
text(ax, 41.0, 27.0, r"$v_1=v_1'\,/\,S(L(v_2))$", size=7.2)
text(ax, 41.0, 24.5, "(gates $S>0$)", size=6.4, color="#555555")
text(ax, 41.0, 20.4, r"Gate $S(x)=e^{2\tanh(ax)}$" + "\n$a$ learned, 0 at the start", size=6.8, linespacing=1.3)
text(ax, 41.0, 14.3, "$L$: Fourier layer\nspectral conv, 48 modes\n(zero-padded by 8)\n+ Conv1d ($k{=}3$),\nGroupNorm, GELU", size=6.8, linespacing=1.3)
text(ax, 41.0, 8.9, "117,796 parameters", size=6.6, color="#555555")

# ---- middle band notes ------------------------------------------------------------------------
arrow(ax, [(3.2, 28.2), (7.4, 28.2)], color=FWD_COL, lw=1.4)
text(ax, 8.6, 28.2, "forward direction: design $\\rightarrow$ ERP", size=7.8, color=FWD_COL, ha="left")
arrow(ax, [(7.4, 25.0), (3.2, 25.0)], color=INV_COL, lw=1.4, ls="--")
text(ax, 8.6, 25.0, "inverse direction: ERP $\\rightarrow$ designs", size=7.8, color=INV_COL, ha="left")
text(ax, 59.6, 29.4, "Only the coupling stack is an exact bijection;\nthe lifts $P,P'$ and readouts $Q,Q'$ are learned maps,\ntied together by the losses.\n"
                     "Several designs per target; one is chosen by the\nmodel's own forward direction (best predicted spectrum).",
     size=6.9, color="#333333", linespacing=1.35)

# ---- loss panel -------------------------------------------------------------------------------
box(ax, 75.0, 44.4, 21.6, 4.4, "Training data: pairs $(\\mathbf{a},y)$", fc=YELLOW, size=8.4)
losses = [
    (38.4, r"Forward loss $J_{FWD}$", r"$\mathrm{MSE}(\hat y,y)$, $\hat y=Q(\mathrm{stack}(P(\mathbf{a})))$"),
    (32.2, r"Inverse loss $J_{INV}$", r"$\mathrm{MSE}(\hat{\mathbf{a}},\mathbf{a})$, $\hat{\mathbf{a}}$ from $y$ via $P'$, stack$^{-1}$, $Q'$"),
    (26.0, r"Round trips $J_{PQ'}$, $J_{P'Q}$", "lift and readout only (stack skipped)"),
    (19.8, r"Cycle $0.1\,J_{cyc}$, alignment $0.1\,J_{align}$", r"$\mathbf{a}\rightarrow\hat y\rightarrow\mathbf{a}$;  $\Vert u_0-v_K\Vert^2/\Vert v_K\Vert^2$"),
    (13.6, r"$\beta$-VAE loss $J_{\beta\mathrm{VAE}}$", r"MSE(recon, $\mathbf{a}$) $+\,\beta\,\mathrm{KL}$, $\beta=0.05$"),
]
mids = []
for y0, name, formula in losses:
    panel(ax, 74.4, y0, 22.4, 5.4, fc=GRAY, r=0.9, z=3)
    text(ax, 85.6, y0 + 3.8, name, size=7.9)
    text(ax, 85.6, y0 + 1.6, formula, size=6.9)
    mids.append(y0 + 2.7)
ax.plot([96.6, 98.0, 98.0], [46.6, 46.6, mids[-1]], lw=1.1, color="black", zorder=6)
for m in mids:
    arrow(ax, [(98.0, m), (96.9, m)], lw=1.0, ms=6)
box(ax, 74.4, 5.2, 22.4, 7.0, "Training in 3 stages (Adam, batch 128)\n1: all but the VAE, 20 epochs, lr $5\\times10^{-4}$\n2: VAE alone, 25 epochs, lr $10^{-3}$\n3: everything together, 15 epochs, lr $3\\times10^{-4}$", fc="#f7f0c8", size=6.8)

text(ax, 50.0, 2.9, "iDCO and iGNO use the same scaffold with the DCO block or the GNO layer as gate $L$ (and a matching lift $P$).  "
                    "Sorted encoder: MLP $250\\rightarrow80\\rightarrow80\\rightarrow48$ on the two resonators' 125 features each.", size=7.8, color="#444444")
text(ax, 50.0, 1.3, "Recommended configuration: bounded design coordinates, bounded gate, binned readout, padded FFT, cycle and alignment terms, VAE trained on the "
                    "stage-1 estimates.  Data: 2 resonators, 18 plate modes.", size=7.8, color="#444444")
save(fig, "invertible_operator")
