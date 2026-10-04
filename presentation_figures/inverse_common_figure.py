"""Common form of the six inverse generators; the conditional generative head is the general, changing block.

Facts: erp_inverse_operators/*.py and the b12 checkpoints in models/100k_2res_grid_18modes (2 resonators, design 8
numbers, 8 samples per target).  Parameter counts from the registry builders (design_dim 8): MDN 92,906, cVAE 148,128,
Flow 209,600, Diffusion 93,640, BasisFlow 206,845, PadINN 810,704.  Adam, cosine to 1 % of lr, 100 epochs, batch 64.
"""
import numpy as np

from common_forms_kit import *  # noqa: F401,F403
from common_forms_kit import elem, card, frame

ex = np.load(OUT / "inverse_example.npz")
fig, ax = new_figure()
rx = frame(ax, 71.2, "Inverse models: common structure",
           r"target ERP spectrum $\rightarrow$ samples of resonator designs; only the generative head changes between models")

Y0, Y1 = 33.2, 48.3
M = (Y0 + Y1) / 2
panel(ax, 2.3, Y0, 11.6, Y1 - Y0, fc=BLUE, r=1.3, z=2)
text(ax, 8.1, Y1 - 1.25, "Target spectrum", size=8.6)
spectrum_inset(fig, ax, 3.5, Y0 + 3.3, 9.2, 7.0, ex["target"], color="black", fill=False)
text(ax, 8.1, Y0 + 1.5, r"$y(f)$, 301 points", size=7.4, color="#333333")
elem(ax, 15.5, 31.2, Y0, Y1, "Spectrum encoder", "3 $\\times$ Conv1d (48 channels,\nkernel 5), SiLU;\nmean + max pooling;\nMLP $\\rightarrow$ embedding\n$e$ (96)", BLUE, size=7.6, tsize=9.0)
elem(ax, 32.8, 52.4, Y0, Y1, "Conditional generative head", "draws designs\n$\\hat{\\mathbf{a}}\\sim p(\\mathbf{a}\\,|\\,y)$\n\n(changes with\nthe model)", ORANGE, size=7.8, tsize=9.0, vary=True)
elem(ax, 54.0, 70.4, Y0, Y1, "Candidate designs", "8 samples $\\hat{\\mathbf{a}}$ per target:\n$[m,f_t,x,y]$ of the two\nresonators, sorted by $f_t$,\nlogit-mapped to the\nphysical ranges\n(then solver check, right)", ORANGE, size=7.4, tsize=9.0)
for xa, xb in ((13.9, 15.5), (31.2, 32.8), (52.4, 54.0)):
    arrow(ax, [(xa, M), (xb, M)], lw=1.2, ms=8)
text(ax, 22.4, 31.7, r"embedding $e$ conditions the head", size=7.0, color="#444444") if False else None
text(ax, 36.1, 31.6, "PadINN works on the raw spectrum instead of the encoder embedding; BasisFlow uses the encoder as its learned basis.", size=6.9, color="#444444")
text(ax, 36.1, 29.9, "Generative head of each model", size=9.6)

CARDS = [
    ("MDN: mixture density network", ["head: $e\\rightarrow$ mixture of 10 diagonal Gaussians over $\\mathbf{a}$", "(weights, means, log-stds); k-means warm start", "sampling: pick a component, draw from it", "loss: mixture negative log-likelihood", "density: exact $\\log p$"], "92,906 parameters"),
    ("cVAE: conditional VAE", ["head: posterior $q(z|\\mathbf{a},e)$, decoder $p(\\mathbf{a}|z,e')$", "(Gaussian, learned variance); latent $z$: 8", "sampling: $z\\sim\\mathcal{N}(0,I)$, decode", "loss: reconstruction NLL + $\\beta$ KL", "($\\beta=0.1$, warm-up over 30 epochs)", "density: $\\log p$ by importance sampling"], "148,128 parameters"),
    ("Flow: conditional flow", ["head: 8 affine coupling layers conditioned on $e$", "(masks alternate, 96 hidden units)", "sampling: $z\\sim\\mathcal{N}(0,I)$ through the inverse", "loss: exact negative log-likelihood", "density: exact $\\log p$"], "209,600 parameters"),
    ("Diffusion: conditional diffusion", ["head: MLP denoiser (128) of noisy design,", "step and $e$", "sampling: 100 reverse steps from noise", "loss: MSE of the predicted noise", "density: none"], "93,640 parameters"),
    ("BasisFlow: basis + flow", ["head: spectrum $\\rightarrow$ Gaussian over 8 coefficients", "(learned basis); coupling flow (8 layers)", "maps coefficients $\\leftrightarrow$ design", "loss: NLL + 0.1 basis reconstruction", "+ 0.1 forward consistency; exact $\\log p$"], "206,845 parameters"),
    ("PadINN: padded invertible net", ["head: one bijection between the zero-padded", "design and [spectrum; $z$], $z$: 16 numbers", "sampling: invert $[y,z]$, $z\\sim\\mathcal{N}(0,I)$", "loss: spectrum MSE + design MSE", "+ 50 $\\times$ MMD($z$); density: none"], "810,704 parameters"),
]
CW, GAP = 21.8, 0.7
for i, (t, lines, par) in enumerate(CARDS):
    r, c = divmod(i, 3)
    card(ax, 2.3 + c * (CW + GAP), 17.2 if r == 0 else 5.1, CW, 11.6, t, lines, par, size=7.9)

# ---- right panel ------------------------------------------------------------------------------
cx = (rx + 99.2) / 2
text(ax, cx, 52.4, "Training and evaluation", size=12.5)
bx0, bx1 = rx + 1.4, 99.2 - 1.6
box(ax, bx0, 44.0, bx1 - bx0, 4.4, "Training data: pairs (spectrum $y$, design $\\mathbf{a}$)", fc=YELLOW, size=7.8)
arrow(ax, [(cx, 44.0), (cx, 42.2)], lw=1.1)
panel(ax, bx0, 31.6, bx1 - bx0, 10.6, fc=GRAY, r=1.0, z=3)
text(ax, cx, 40.4, "Loss", size=8.8)
text(ax, cx, 36.6, "depends on the head\n(see the cards): likelihood,\nELBO, noise MSE or\nMSE + MMD", size=7.4, linespacing=1.3)
arrow(ax, [(cx, 31.6), (cx, 30.2)], lw=1.1)
box(ax, bx0, 22.2, bx1 - bx0, 7.8, "Training, all six:\nAdam, cosine decay, 100 epochs, batch 64\nlr $5\\times10^{-4}$ (Flow, BasisFlow, PadINN)\nlr $10^{-3}$ (MDN, cVAE, Diffusion)", fc="#f7f0c8", size=7.0)
panel(ax, bx0, 5.2, bx1 - bx0, 15.6, fc=PINK, r=1.0, z=3)
text(ax, cx, 19.0, "Evaluation: solver check", size=8.6)
text(ax, cx, 15.9, "plate solver gives the ERP of each\nof the 8 candidates; one is selected:", size=7.2, linespacing=1.3)
text(ax, bx0 + 1.0, 12.0, "random: the first sample", size=7.2, ha="left")
text(ax, bx0 + 1.0, 10.2, "own: highest model $\\log p$", size=7.2, ha="left")
text(ax, bx0 + 1.0, 8.6, "(none for Diffusion, PadINN)", size=6.6, ha="left", color="#555555")
text(ax, bx0 + 1.0, 6.7, "oracle: lowest solver error", size=7.2, ha="left")
text(ax, 50.0, 2.9, "Design $\\mathbf{a}$: the two resonators sorted by $f_t$, each $[m,f_t,x,y]$, mapped by a logit to unbounded coordinates (bounded design space; $k=m(2\\pi f_t)^2$).", size=7.6, color="#444444")
text(ax, 50.0, 1.3, "Data: 100k configurations, 2 resonators on a $14\\times5$ position grid, 18 plate modes, 80/10/10 split.  Spectrum encoder: shared by MDN, cVAE (two copies), Flow, Diffusion.", size=7.6, color="#444444")
save(fig, "inverse_models_common")
