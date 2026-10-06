"""Presentation figure: inverse model (conditional normalizing flow, Flow_b12) and its loss / evaluation.

Numbers read from the code and the trained checkpoint erp_inverse/models/100k_2res_grid_18modes/
inverse_flow_b12.pth: 209,600 parameters (spectrum encoder 42,048 + 8 couplings 167,552), Adam 5e-4, cosine
(eta_min 0.01 lr), 100 epochs, batch 64, grad-clip 5, 2 resonators -> D = 8, 8 samples per target.
The spectrum and the candidate designs come from a real held-out target (make_inverse_example.py).
"""
import numpy as np

from diagram_kit import *  # noqa: F401,F403
from diagram_kit import BLUE, GREEN, ORANGE, YELLOW, GRAY, PINK, OUT

ex = np.load(OUT / "inverse_example.npz")
fig, ax = new_figure()

panel(ax, 1.0, 4.4, 63.8, 49.6, fc="#f6f6f6", ls="--", lw=1.2, r=2.0, z=0)
panel(ax, 66.0, 4.4, 33.2, 49.6, fc="#f6f6f6", ls="--", lw=1.2, r=2.0, z=0)
text(ax, 32.9, 52.4, "Inverse model: conditional normalizing flow", size=12.5)
text(ax, 32.9, 50.6, r"target ERP spectrum $\rightarrow$ samples of resonator designs;  209,600 parameters", size=8.5, color="#555555")
text(ax, 82.6, 52.4, "Loss and evaluation", size=12.5)

# ---- top row: target spectrum -> spectrum encoder -> embedding e ------------------------------
panel(ax, 2.3, 32.0, 14.2, 15.5, fc=BLUE, r=1.4)
text(ax, 9.4, 46.2, "Target spectrum", size=9)
spectrum_inset(fig, ax, 3.8, 36.4, 11.4, 7.6, ex["target"], color="black", fill=False)
text(ax, 9.4, 33.9, r"$y(f)$, 301 frequencies", size=7.4, color="#333333")
arrow(ax, [(16.5, 39.7), (19.3, 39.7)], lw=1.0)

panel(ax, 19.3, 32.0, 23.2, 15.5, fc=BLUE, r=1.4)
text(ax, 30.9, 46.2, "Spectrum encoder", size=9)
xs = [24.5, 30.9, 37.3]
mlp(ax, xs, [[43.2, None, 37.6]] * 3, size=7.2, r=0.85)
for x in xs:
    dots(ax, x, 40.4, size=8)
text(ax, 30.9, 34.0, "3 $\\times$ Conv1d (48 channels, kernel 5), SiLU;\nmean + max pooling; MLP $96\\rightarrow96\\rightarrow96$",
     size=6.6, color="#333333", linespacing=1.2)
arrow(ax, [(42.5, 39.7), (44.8, 39.7)], lw=1.0)

panel(ax, 44.8, 32.0, 5.0, 15.5, fc=BLUE, r=1.4)
text(ax, 47.3, 46.2, r"$\mathbf{e}$", size=9.5)
neuron(ax, 47.3, 43.0, r"$e_1$", r=1.05, size=6.6)
dots(ax, 47.3, 40.0, size=8)
neuron(ax, 47.3, 37.0, r"$e_{96}$", r=1.05, size=6.2)
text(ax, 47.3, 33.9, "96", size=6.8, color="#333333")

text(ax, 51.8, 46.3, r"Design $\mathbf{a}$ (8 numbers):", size=7.4, ha="left")
text(ax, 51.8, 45.0, "$[m,f_t,x,y]$ of the two\nresonators, sorted by $f_t$,\nmapped by a logit to\nunbounded coordinates, so\nevery sample is a valid\ndesign; $k=m(2\\pi f_t)^2$",
     size=7.0, ha="left", va="top", color="#222222", linespacing=1.35)

# ---- bottom row: z -> conditional flow -> candidate designs -------------------------------------
panel(ax, 2.3, 7.0, 8.2, 20.0, fc=GREEN, r=1.4)
text(ax, 6.4, 25.8, r"$\mathbf{z}$", size=10)
text(ax, 6.4, 24.0, r"$\mathcal{N}(0,I)$", size=7.4)
neuron(ax, 6.4, 20.6, r"$z_1$", r=1.05, size=6.8)
dots(ax, 6.4, 17.6, size=8)
neuron(ax, 6.4, 14.6, r"$z_8$", r=1.05, size=6.8)
text(ax, 6.4, 9.6, "8 numbers", size=6.8, color="#333333")

panel(ax, 12.8, 7.0, 32.7, 20.0, fc=GREEN, r=1.4)
text(ax, 29.15, 25.9, "Conditional flow: 8 affine coupling layers (sampling pass)", size=8.8)
BY0, BY1 = 15.2, 21.2
BM = (BY0 + BY1) / 2
blocks = [(14.2, r"$T_8^{-1}$"), (23.2, r"$T_7^{-1}$"), (37.2, r"$T_1^{-1}$")]
for x0, lab in blocks:
    box(ax, x0, BY0, 6.6, BY1 - BY0, lab, fc="white", size=9.5)
arrow(ax, [(10.5, BM), (14.2, BM)], lw=1.1)
arrow(ax, [(20.8, BM), (23.2, BM)], lw=1.1)
arrow(ax, [(29.8, BM), (37.2, BM)], lw=1.1)
text(ax, 33.5, BM + 1.3, r"$\cdots$", size=11)
arrow(ax, [(43.8, BM), (48.0, BM)], lw=1.1)
# conditioning bus from the embedding
BUS = 23.6
ax.plot([47.3, 47.3, 17.5], [32.0, BUS, BUS], lw=1.0, color="black", zorder=6)
for x0, _ in blocks:
    arrow(ax, [(x0 + 3.3, BUS), (x0 + 3.3, BY1 + 0.15)], lw=1.0, ms=7)
text(ax, 33.5, BUS + 1.0, r"conditioning on $\mathbf{e}$", size=6.8, color="#333333")
text(ax, 29.15, 12.7, "$T_\\ell$: $a\\leftarrow a\\odot m+(1-m)\\odot(a\\,e^{\\log s}+t)$, masks $m$ alternate even / odd\n"
                      "$(\\log s,t)$ = MLP$([a\\odot m,\\mathbf{e}])$, 96 hidden units, $\\log s\\in[-2,2]$", size=6.6, color="#333333", linespacing=1.3)
arrow(ax, [(43.2, 10.2), (14.6, 10.2)], lw=1.0, ls="--", color="#444444")
text(ax, 29.15, 8.5, r"training pass: $\mathbf{a}\rightarrow\mathbf{z}$ with the exact $\log p(\mathbf{a}\,|\,y)$", size=6.8, color="#333333")

panel(ax, 48.0, 7.0, 15.5, 20.0, fc=ORANGE, r=1.4)
text(ax, 55.75, 25.8, "Candidate designs", size=9)
PX0, PY0, PW, PH = 49.2, 17.4, 13.1, 4.7
panel(ax, PX0, PY0, PW, PH, fc="#fffaf3", ec=EDGE, lw=0.9, r=0.4, z=3)
sx = lambda x: PX0 + x / 1.4 * PW
sy = lambda y: PY0 + y / 0.5 * PH
ax.plot([sx(0.865)], [sy(0.309)], marker="*", color="#35d0ff", ms=8, mec="black", mew=0.6, ls="", zorder=7)
for d in ex["designs"]:
    ax.plot([sx(d[0, 3])], [sy(d[0, 4])], marker="o", color="#2a78d6", ms=3.6, mec="black", mew=0.3, alpha=0.8, ls="", zorder=8)
    ax.plot([sx(d[1, 3])], [sy(d[1, 4])], marker="s", color="#eb6834", ms=3.6, mec="black", mew=0.3, alpha=0.8, ls="", zorder=8)
tc = ex["true_cfg"]
for r_, mk in ((0, "o"), (1, "s")):
    ax.plot([sx(tc[r_, 3])], [sy(tc[r_, 4])], marker=mk, mfc="none", mec="black", mew=1.2, ms=7.5, ls="", zorder=9)
text(ax, 55.75, 15.4, "8 samples $\\hat{\\mathbf{a}}^{(n)}$ for one real target\n(plate $1.4\\,\\mathrm{m}\\times0.5\\,\\mathrm{m}$)", size=6.8, color="#333333", linespacing=1.25)
ax.plot([49.9], [12.4], marker="o", color="#2a78d6", ms=3.8, mec="black", mew=0.3, ls="", zorder=8)
text(ax, 50.8, 12.4, "resonator with lower $f_t$", size=6.6, ha="left")
ax.plot([49.9], [10.9], marker="s", color="#eb6834", ms=3.8, mec="black", mew=0.3, ls="", zorder=8)
text(ax, 50.8, 10.9, "resonator with higher $f_t$", size=6.6, ha="left")
ax.plot([49.9], [9.4], marker="o", mfc="none", mec="black", mew=1.1, ms=5.5, ls="", zorder=8)
text(ax, 50.8, 9.4, "true design of the target", size=6.6, ha="left")

# ---- loss and evaluation ----------------------------------------------------------------------
box(ax, 70.0, 44.8, 25.0, 4.2, "Training data: pairs (spectrum $y$, design $\\mathbf{a}$)", fc=YELLOW, size=8.6)
arrow(ax, [(82.5, 44.8), (82.5, 42.7)], lw=1.1)
panel(ax, 68.5, 34.2, 28.0, 8.4, fc=GRAY, r=1.0, z=3)
text(ax, 82.5, 40.7, "Negative log-likelihood loss", size=9.2)
text(ax, 82.5, 38.5, r"$\mathcal{L}=-\log p(\mathbf{a}\,|\,y)$", size=9.6)
text(ax, 82.5, 36.0, r"$=\frac{1}{2}\Vert\mathbf{z}\Vert^2+\frac{D}{2}\log2\pi-\sum_{\ell,i}\log s_{\ell,i}$,  $D=8$", size=8.6)
arrow(ax, [(82.5, 34.2), (82.5, 32.8)], lw=1.1)
diamond(ax, 82.5, 30.3, 17.0, 5.0, r"$\mathrm{arg\,min}_\theta\,\mathcal{L}(\theta)$", size=9.5)
text(ax, 82.5, 26.3, "Adam ($5\\times10^{-4}$), cosine decay, 100 epochs, batch 64", size=6.9, color="#555555")
arrow(ax, [(74.0, 30.3), (51.5, 30.3)], lw=1.1)
text(ax, 58.0, 31.4, r"update $\theta$ (encoder and flow)", size=7.2, color="#333333")

panel(ax, 68.5, 7.4, 28.0, 17.0, fc=PINK, r=1.0, z=3)
text(ax, 82.5, 22.5, "Evaluation: solver check of the 8 candidates", size=8.6)
text(ax, 82.5, 19.6, "plate solver $\\rightarrow$ ERP of each candidate,\ncompared with the target spectrum", size=7.6, linespacing=1.3)
text(ax, 70.0, 15.6, "One candidate is selected by:", size=7.6, ha="left")
text(ax, 70.6, 13.7, "random: the first sample", size=7.4, ha="left")
text(ax, 70.6, 12.0, r"own: highest model $\log p$", size=7.4, ha="left")
text(ax, 70.6, 10.3, "oracle: lowest solver error (uses the target)", size=7.4, ha="left")
arrow(ax, [(63.5, 17.0), (68.5, 17.0)], lw=1.1)

text(ax, 50.0, 2.9, "Design $\\mathbf{a}$: the two resonators sorted by $f_t$, each $[m,f_t,x,y]$, mapped by a logit to unbounded coordinates (bounded design space). "
                    "Data: 100k configurations, 2 resonators on a $14\\times5$ position grid, 18 plate modes, 80/10/10 split.", size=8.0, color="#444444")
text(ax, 50.0, 1.3, "Other generators trained on the same data: MDN (10 Gaussians), cVAE (latent 8), Diffusion (100 steps), BasisFlow, PadINN.", size=8.0, color="#444444")
save(fig, "inverse_model")
