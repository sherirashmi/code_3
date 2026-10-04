"""Common form of the Invertible DeepONet (Kaltenbach et al.); the trunk is the general, changing block.

Facts: erp_invertible_deeponet/model.py and the nine checkpoints in models/200k_2res_18modes.
Parameters (parameters only, buffers excluded): Q8 938,665; Q64 1,384,257; Q128 1,893,505; Q64-FNO 1,388,945;
Q64-DCO 1,402,049; Q64-DNO 1,325,377; Q64-WNO 1,319,665; Q64-LNO 1,240,321; Q64-SIREN 1,310,273 (RealNVP 1,153,280 for Q64).
Training: Adam + cosine, lr 5e-4, 150 epochs (early stop), batch 128, inverse weight ramps over 10 epochs.
"""
from common_forms_kit import *  # noqa: F401,F403
from common_forms_kit import elem, card, frame, loss_panel, FWD_COL, INV_COL, VARY_FILL, VARY_EDGE

fig, ax = new_figure()
rx = frame(ax, 71.2, "Invertible DeepONet: common structure",
           r"design $\leftrightarrow$ ERP through a bijection (RealNVP) and basis functions of $f$; only the trunk changes between variants")

# ---- forward row ------------------------------------------------------------------------------
FY0, FY1 = 40.2, 48.4
FM = (FY0 + FY1) / 2
elem(ax, 2.3, 9.3, FY0, FY1, r"Design $\mathbf{a}$", "$D=8$\n(2 resonators)", BLUE, size=7.4, tsize=8.6)
elem(ax, 10.9, 21.4, FY0, FY1, "Pad", "$Q-D$ zeros\nQ8: 0, Q64: 56,\nQ128: 120", BLUE, size=7.2, tsize=8.6)
elem(ax, 23.0, 39.0, FY0, FY1, r"RealNVP $T$", "10 affine couplings\n(2 hidden layers $\\times$ 256)\nbijection on $\\mathbb{R}^Q$", BLUE, size=7.2, tsize=8.6)
elem(ax, 40.6, 48.0, FY0, FY1, "Coefficients", "$\\mathbf{b}=T(\\cdot)$\n$Q$ numbers", BLUE, size=7.2, tsize=8.4)
CX, CYc = 51.2, FM
circle_op(ax, CX, CYc, r"$\cdot$", r=1.4, size=12)
panel(ax, 54.0, FY0, 16.4, FY1 - FY0, fc=ORANGE, r=1.3, z=2)
text(ax, 62.2, FY1 - 1.2, "Predicted ERP", size=8.4)
spectrum_inset(fig, ax, 56.0, FY0 + 1.0, 5.4, 3.8, sample_erp())
text(ax, 66.6, FM - 0.6, "$\\hat y=\\Psi\\mathbf{b}+\\psi_0$", size=7.6)
for xa, xb in ((9.3, 10.9), (21.4, 23.0), (39.0, 40.6)):
    arrow(ax, [(xa, FM), (xb, FM)], color=FWD_COL, lw=1.3, ms=8)
arrow(ax, [(48.0, FM), (CX - 1.4, FM)], color=FWD_COL, lw=1.3, ms=8)
arrow(ax, [(CX + 1.4, FM), (54.0, FM)], color=FWD_COL, lw=1.3, ms=8)

# ---- trunk slot (general, changing block) ------------------------------------------------------
panel(ax, 2.3, 29.6, 58.7, 8.8, fc=VARY_FILL, ec=VARY_EDGE, ls="--", lw=1.7, r=1.3, z=2)
text(ax, 3.4, 37.2, "Trunk: basis functions of the frequency (changes with the variant)", size=8.4, ha="left")
box(ax, 3.4, 30.6, 8.0, 4.6, "frequency $f$\n(301 points)", fc="white", size=6.9)
box(ax, 13.4, 30.6, 24.6, 4.6, "trunk network $\\rightarrow$ $Q+1$ functions of $f$\n(variants below)", fc="white", size=7.2)
box(ax, 40.0, 30.6, 19.8, 4.6, "QR: $\\Psi$ ($F\\times Q$), $\\Psi^\\top\\Psi=F\\,I$;\nbias function $\\psi_0$", fc="white", size=6.9)
arrow(ax, [(11.4, 32.9), (13.4, 32.9)], lw=1.1, ms=7)
arrow(ax, [(38.0, 32.9), (40.0, 32.9)], lw=1.1, ms=7)
arrow(ax, [(CX, 35.2), (CX, 38.4)], color=FWD_COL, lw=1.3, ms=8) if False else None
ax.plot([CX, CX], [35.2, 38.4], color=FWD_COL, lw=1.3, zorder=6)
arrow(ax, [(CX, 38.4), (CX, CYc - 1.4)], color=FWD_COL, lw=1.3, ms=8)

# ---- inverse row ------------------------------------------------------------------------------
IY0, IY1 = 21.0, 28.0
IM = (IY0 + IY1) / 2
elem(ax, 62.0, 70.4, IY0, IY1, "Target $y$", "ERP (301)", YELLOW, size=7.2, tsize=8.4)
elem(ax, 45.6, 60.4, IY0, IY1, "Projection", "$\\mathbf{b}^*=\\Psi^\\top(y-\\psi_0)/F$\n(closed form, exact)", ORANGE, size=7.0, tsize=8.4)
elem(ax, 30.4, 44.0, IY0, IY1, "Samples", "$\\mathbf{b}\\sim\\mathcal{N}(\\mathbf{b}^*,\\,s^2/F\\cdot I)$\n(sample 0: $\\mathbf{b}^*$)", ORANGE, size=6.9, tsize=8.4)
elem(ax, 16.6, 28.8, IY0, IY1, r"RealNVP$^{-1}$", "$T^{-1}\\rightarrow[\\mathbf{a},\\mathbf{z}]$", ORANGE, size=7.2, tsize=8.4)
elem(ax, 2.3, 15.0, IY0, IY1, r"Designs $\hat{\mathbf{a}}$", "latent $\\mathbf{z}$ dropped\n(several samples)", ORANGE, size=6.9, tsize=8.4)
for xa, xb in ((62.0, 60.4), (45.6, 44.0), (30.4, 28.8), (16.6, 15.0)):
    arrow(ax, [(xa, IM), (xb, IM)], color=INV_COL, lw=1.3, ls="--", ms=8)
text(ax, 36.1, 19.5, "Trunk variants: Q = 64 with the same RealNVP branch (1,153,280 parameters)", size=9.0)

TR = [
    ("MLP trunk", ["Fourier features", "(32 frequencies)", "$\\rightarrow$ MLP, 4 layers", "$\\times$ 256, SiLU"], "trunk 230,977", "total 1,384,257"),
    ("FNO trunk", ["Fourier features", "$\\rightarrow$ lift (40)", "$\\rightarrow$ 4 Fourier blocks", "(32 modes, padded)", "$\\rightarrow$ MLP"], "trunk 235,665", "total 1,388,945"),
    ("DCO trunk", ["Fourier features", "$\\rightarrow$ lift (128)", "$\\rightarrow$ 4 residual", "MLP blocks $\\rightarrow$ local", "refinement $\\rightarrow$ linear"], "trunk 248,769", "total 1,402,049"),
    ("DNO trunk", ["Fourier features", "$\\rightarrow$ lift (96)", "$\\rightarrow$ 4 FiLM blocks", "(FiLM from $f$)", "$\\rightarrow$ refinement"], "trunk 172,097", "total 1,325,377"),
    ("WNO trunk", ["Fourier features", "$\\rightarrow$ lift (40)", "$\\rightarrow$ 4 wavelet", "blocks (3-level", "Haar) $\\rightarrow$ MLP"], "trunk 166,385", "total 1,319,665"),
    ("LNO trunk", ["48 learned poles", "and residues (the", "same for every design)", "+ Fourier features", "$\\rightarrow$ lift (96) $\\rightarrow$ MLP"], "trunk 87,041", "total 1,240,321"),
    ("SIREN trunk", ["$f\\rightarrow$ 4 sine layers", "(128, $\\omega_0=10$)", "$\\rightarrow$ local", "refinement", "$\\rightarrow$ linear"], "trunk 156,993", "total 1,310,273"),
]
CW, GAP = 9.1, 0.45
for i, (t, lines, p1, p2) in enumerate(TR):
    x = 2.3 + i * (CW + GAP)
    panel(ax, x, 5.1, CW, 13.4, fc=VARY_FILL, ec=VARY_EDGE, lw=1.1, r=0.9, z=2)
    text(ax, x + CW / 2, 17.4, t, size=8.2)
    ax.text(x + 0.5, 15.8, "\n".join(lines), fontsize=7.5, ha="left", va="top", zorder=10, linespacing=1.4, color="#222222")
    text(ax, x + CW / 2, 7.4, p1, size=6.2, color="#555555")
    text(ax, x + CW / 2, 6.0, p2, size=6.2, color="#555555")

loss_panel(ax, rx, [
    ("Forward loss", r"MSE$(\hat y,\,y)$, normalised ERP"),
    ("Inverse loss", r"Huber$(\hat{\mathbf{a}},\mathbf{a})$, $\hat{\mathbf{a}}=T^{-1}(\mathbf{b}^*(y))$"),
    (r"Latent loss ($\times\,0.1$)", r"Huber$(\hat{\mathbf{z}},0)$, only if pad $>0$"),
], "Adam, cosine decay, lr $5\\times10^{-4}$,\n150 epochs, batch 128;\ninverse weight ramps up over\nthe first 10 epochs;\nbest validation weights kept.\nQ8 and Q64-SIREN stopped at\nepoch 50 (best: 37, 40).\nAfterwards $s^2$ = training\nforward residual.", data_label="Training data: pairs ($y$, $\\mathbf{a}$)", box_h=6.0, footer_size=7.2)
text(ax, 50.0, 2.9, "A trunk must not see the design, otherwise the inverse is no longer a projection.  Size variants with the MLP trunk: Q8 (938,665 parameters), Q64 (1,384,257), Q128 (1,893,505).", size=7.6, color="#444444")
text(ax, 50.0, 1.3, "Data: 200k configurations, 2 resonators ($m,f_t,x,y$ by Latin hypercube, $k=m(2\\pi f_t)^2$), 18 plate modes.  Design in bounded logit coordinates, sorted by $f_t$.", size=7.6, color="#444444")
save(fig, "invertible_deeponet_common")
