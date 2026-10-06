"""Invertible DeepONet (iDON), layer by layer, from erp_invertible_deeponet/scripts/model.py and the checkpoints in
erp_invertible_deeponet/models/200k_2res_18modes: the sizes below are those of Q64 (D = 8 design numbers, 56 padding zeros, Q = 64, F = 301 frequencies).

  idon_arch_forward        forward pass: branch (10 affine couplings), trunk (Fourier features + 4 x 256 MLP), QR, combination ERP = Psi b + psi0
  idon_arch_coupling       one affine coupling layer (forward and exact inverse), with its MLP
  idon_arch_inverse        inverse pass: projection, sampling, T^-1, design decoding
  idon_arch_table          every layer with input -> output size and parameter count, hyper-parameters, training, results
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_kit import (circle_op, BLUE, EDGE, GRAY, GREEN, ORANGE, PINK, PURPLE, YELLOW, W, H, arrow, box, dots, new_figure, neuron,  # noqa: E402
                         panel, save, spectrum_inset, text, titled_panel)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CURVE = np.asarray(json.load(open(Path(__file__).resolve().parent / "erp_curve.json")), dtype=float)
FWD, INV = "#1f5fbf", "#d9622b"
R = 0.8


def column(ax, x, ys, labels=None, fc="#ededed", size=6.5, r=R):
    """Neurons at ys; a None in ys is drawn as vertical dots."""
    drawn = [y for y in ys if y is not None]
    for i, y in enumerate(ys):
        if y is None:
            above = ys[i - 1]
            below = ys[i + 1]
            dots(ax, x, (above + below) / 2, size=9)
            continue
        neuron(ax, x, y, labels[i] if labels else "", r=r, fc=fc, size=size)
    return drawn


def links(ax, x0, ys0, x1, ys1, alpha=0.4, r=R):
    for y0 in [y for y in ys0 if y is not None]:
        for y1 in [y for y in ys1 if y is not None]:
            ax.plot([x0 + r, x1 - r], [y0, y1], color="#555555", lw=0.45, alpha=alpha, zorder=2)


def forward_figure():
    fig, ax = new_figure(20)
    text(ax, 50, 55.0, "Invertible DeepONet (Q64): forward pass, design $a$ $\\rightarrow$ ERP", size=15)

    # ---------------- branch ----------------
    titled_panel(ax, 1, 28.2, 77, 25.3, "", BLUE)
    text(ax, 2.2, 51.8, "Branch: design $\\rightarrow$ coefficients $b = T(a)$, T = RealNVP (bijection)", size=12.5, ha="left")
    # input a: 8 nodes
    ys_a = [46.5 - 1.9 * i for i in range(8)]  # last node at 33.2
    column(ax, 7, ys_a, ["$m_1$", "$f_1$", "$x_1$", "$y_1$", "$m_2$", "$f_2$", "$x_2$", "$y_2$"], fc="#cfe0f7", size=6)
    text(ax, 7, 48.9, "design $a$", size=10.5)
    text(ax, 7, 30.4, "8 nodes: 2 $\\times$ [m, $f_t$, x, y]\n(bounded-logit coordinates)", size=7.8, linespacing=1.25)
    # padded vector: 64 nodes (8 design + 56 zeros)
    ys_p = [46.5, 44.6, 42.7, 40.8, None, 37.0, 35.1, 33.2]
    xp = 17
    column(ax, xp, ys_p, ["$a_1$", "$a_2$", "$a_3$", "$a_4$", None, "0", "0", "0"], fc="#cfe0f7", size=6)
    text(ax, xp, 48.9, "$[a,\\ z=0]$", size=10.5)
    text(ax, xp, 30.4, "64 nodes\n= 8 design + 56 zeros", size=7.8, linespacing=1.25)
    arrow(ax, [(8.3, 40.0), (15.7, 40.0)], color=FWD)
    text(ax, 12.0, 41.7, "pad", size=8.5)
    # 10 couplings
    x0, bw, gap = 24.0, 2.35, 0.3
    for i in range(10):
        panel(ax, x0 + i * (bw + gap), 33.0, bw, 14.0, fc="#fbe5d6", r=0.4, z=3, lw=0.9)
        text(ax, x0 + i * (bw + gap) + bw / 2, 40.0, str(i + 1), size=8, rotation=90)
    xs_end = x0 + 10 * (bw + gap) - gap
    text(ax, (x0 + xs_end) / 2, 48.7, "10 affine coupling layers", size=10.5)
    text(ax, (x0 + xs_end) / 2, 30.4, "each 64 $\\rightarrow$ 64: keep 32 numbers, transform the other 32\n$y = x_A \\mid x_B\\,e^{s(x_A)} + t(x_A)$,  (s, t) from an MLP 64-256-256-128",
         size=7.6, linespacing=1.3)
    arrow(ax, [(19.0, 40.0), (23.4, 40.0)], color=FWD)
    # b
    xb = 60
    ys_b = ys_p
    column(ax, xb, ys_b, ["$b_1$", "$b_2$", "$b_3$", "$b_4$", None, "", "", ""], fc="#fbd9c2", size=6)
    for y in (36.0, 33.9, 31.8):
        pass
    text(ax, xb, 48.9, "coefficients $b$", size=10.5)
    text(ax, xb, 30.4, "64 nodes", size=7.8)
    arrow(ax, [(xs_end + 0.7, 40.0), (xb - 1.4, 40.0)], color=FWD)

    # ---------------- trunk ----------------
    titled_panel(ax, 1, 1.2, 77, 26.2, "", GREEN)
    text(ax, 2.2, 25.7, "Trunk: frequency $\\rightarrow$ basis functions $\\Psi(f)$ (does not see the design)", size=12.5, ha="left")
    xs = [6.5, 15.5, 25.5, 34.5, 43.5, 52.5, 61.5]
    names = ["frequency $f$", "Fourier features", "hidden 1", "hidden 2", "hidden 3", "hidden 4", "output"]
    counts = ["1 node\n(each of the 301\ngrid frequencies)", "65 nodes\n$[f,\\ \\sin k\\pi f,\\ \\cos k\\pi f]$,\n$k = 1 \\ldots 32$", "256", "256", "256", "256", "65 nodes\n= 1 ($\\psi_0$) + 64 (basis)"]
    cols = [[13.0], [19.0, 16.5, 14.0, None, 9.0, 6.5], [19.0, 16.5, 14.0, None, 9.0, 6.5], [19.0, 16.5, 14.0, None, 9.0, 6.5],
            [19.0, 16.5, 14.0, None, 9.0, 6.5], [19.0, 16.5, 14.0, None, 9.0, 6.5], [19.0, 16.5, 14.0, None, 9.0, 6.5]]
    for a, b, xa, xb_ in zip(cols[:-1], cols[1:], xs[:-1], xs[1:]):
        links(ax, xa, a, xb_, b)
    fcs = ["#cfe9cf", "#cfe9cf", "#e5f3e5", "#e5f3e5", "#e5f3e5", "#e5f3e5", "#cfe9cf"]
    for x, ys, fc, nm, cnt in zip(xs, cols, fcs, names, counts):
        column(ax, x, ys, None, fc=fc)
        text(ax, x, 22.3, nm, size=9.5)
        text(ax, x, 3.7, cnt, size=7.4, linespacing=1.2)
    ops = ["Fourier\nfeatures", "Linear\n+ SiLU", "Linear\n+ SiLU", "Linear\n+ SiLU", "Linear\n+ SiLU", "Linear"]
    for xa, xb_, o in zip(xs[:-1], xs[1:], ops):
        text(ax, (xa + xb_) / 2, 11.8, o, size=7.4, linespacing=1.15, color="#1d5c1d",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    # split + QR
    box(ax, 66.5, 5.0, 10.2, 15.0, "split\n$\\psi_0$ (1)\n$\\tilde\\Psi$ (64)\n\nQR:\n$\\Psi^\\top\\Psi = F\\,I$\n$\\Rightarrow\\ \\Psi$\n(301 $\\times$ 64)", fc=YELLOW, size=8)
    arrow(ax, [(62.5, 13.0), (66.4, 13.0)], color=FWD)

    # ---------------- combination ----------------
    titled_panel(ax, 79.5, 1.2, 19.5, 52.3, "", YELLOW)
    text(ax, 89.25, 51.6, "Combine (linear in $b$)", size=11.5)
    text(ax, 89.25, 46.0, "$\\mathrm{ERP}(f)=\\sum_q b_q\\,\\psi_q(f)+\\psi_0(f)$", size=10)
    text(ax, 89.25, 42.8, "$q = 1 \\ldots 64,\\ \\ f$ on the 301-point grid", size=8)
    circle_op(ax, 89.25, 33.0, "$\\Psi\\,b+\\psi_0$", r=4.2, size=9)
    arrow(ax, [(63.6, 40.0), (82.0, 40.0), (82.0, 34.3), (85.0, 34.3)], color=FWD)
    text(ax, 72.8, 41.4, "$b$ (64)", size=9)
    arrow(ax, [(76.9, 12.5), (81.8, 12.5), (81.8, 31.5), (85.0, 31.5)], color=FWD)
    text(ax, 80.5, 21.5, "$\\Psi$ (301 $\\times$ 64),  $\\psi_0$ (301)", size=8, rotation=90)
    arrow(ax, [(89.25, 28.8), (89.25, 16.8)], color=FWD)
    text(ax, 90.0, 22.5, "ERP:\n301 points", size=9.5, ha="left", linespacing=1.2)
    spectrum_inset(fig, ax, 84.5, 4.5, 12.5, 10.5, CURVE, color=FWD)
    text(ax, 89.25, 2.6, "10-160 Hz", size=7.5)
    return fig



def coupling_figure():
    fig, ax = new_figure(19)
    text(ax, 50, 54.6, "One affine coupling layer of $T$ (Q64: 64 $\\rightarrow$ 64; the 10 layers have the same form, different masks)", size=14)
    # input x
    titled_panel(ax, 1, 8, 12, 43, "", BLUE)
    text(ax, 7, 49.0, "input $x$\n(64 nodes)", size=11, linespacing=1.2)
    ys_A = [43.0, 40.5, 38.0, None, 33.0]
    ys_B = [24.0, 21.5, 19.0, None, 14.0]
    column(ax, 7, ys_A, ["", "", "", None, ""], fc="#cfe0f7")
    column(ax, 7, ys_B, ["", "", "", None, ""], fc="#fbd9c2")
    text(ax, 7, 46.0, "$x_A$: 32 kept (mask = 1)", size=9.5, color="#1f5fbf")
    text(ax, 7, 28.2, "$x_B$: 32 transformed (mask = 0)", size=9.5, color="#b5532a")
    # masked input to the MLP
    xs = [22.0, 34.5, 47.0, 59.5]
    cols = [[43.0, 40.5, 38.0, None, 33.0, 30.0, 27.5, 25.0, None, 20.0], [44.0, 41.0, 38.0, None, 22.0, 19.0, 16.0],
            [44.0, 41.0, 38.0, None, 22.0, 19.0, 16.0], [44.0, 41.0, 38.0, None, 30.0, 27.0, None, 20.0, 17.0, 14.0]]
    for a, b, xa, xb_ in zip(cols[:-1], cols[1:], xs[:-1], xs[1:]):
        links(ax, xa, a, xb_, b, alpha=0.3)
    column(ax, xs[0], cols[0], None, fc="#e8e8e8")
    for y in (43.0, 40.5, 38.0, 33.0):
        neuron(ax, xs[0], y, "", r=R, fc="#cfe0f7")
    for y in (30.0, 27.5, 25.0, 20.0):
        neuron(ax, xs[0], y, "0", r=R, fc="#f2f2f2", size=6)
    column(ax, xs[1], cols[1], None, fc="#e5f3e5")
    column(ax, xs[2], cols[2], None, fc="#e5f3e5")
    column(ax, xs[3], cols[3], None, fc="#e5f3e5")
    for x, nm, c in zip(xs, ["$x\\odot$mask", "hidden 1", "hidden 2", "output"], ["64 nodes\n($x_A$, zeros)", "256", "256", "128 nodes"]):
        text(ax, x, 48.8, nm, size=11)
        text(ax, x, 10.0, c, size=9.5, linespacing=1.2)
    for xa, xb_, o in zip(xs[:-1], xs[1:], ["Linear\n+ SiLU", "Linear\n+ SiLU", "Linear"]):
        text(ax, (xa + xb_) / 2, 30.5, o, size=9, color="#1d5c1d", linespacing=1.15, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    arrow(ax, [(13.2, 38.0), (20.2, 38.0)], color=FWD)
    text(ax, 16.7, 39.6, "$\\times$ mask", size=9.5)
    # split outputs
    box(ax, 64.0, 36.0, 7.0, 9.0, "$\\log s$\n(64)", fc="#e2d4f0", size=10.5)
    box(ax, 64.0, 13.0, 7.0, 9.0, "$t$\n(64)", fc="#e2d4f0", size=10.5)
    arrow(ax, [(61.0, 38.5), (63.9, 40.5)], color=FWD)
    arrow(ax, [(61.0, 20.0), (63.9, 18.0)], color=FWD)
    text(ax, 67.5, 47.3, "first 64 outputs", size=9)
    text(ax, 67.5, 10.5, "last 64 outputs", size=9)
    box(ax, 74.0, 36.0, 10.0, 9.0, "$s=\\tanh(\\log s)$\n(clamp $\\pm 1$)\n$\\Rightarrow e^{s}\\in[e^{-1},e]$", fc=YELLOW, size=9.2)
    arrow(ax, [(71.1, 40.5), (73.9, 40.5)], color=FWD)
    circle_op(ax, 80.0, 28.5, "$\\times$", r=1.5, size=12)
    circle_op(ax, 88.0, 28.5, "$+$", r=1.5, size=12)
    arrow(ax, [(79.0, 36.0), (80.0, 30.1)], color=FWD)
    arrow(ax, [(81.6, 28.5), (86.4, 28.5)], color=FWD)
    arrow(ax, [(71.1, 17.5), (88.0, 17.5), (88.0, 26.9)], color=FWD)
    # x_B enters the multiplication from the left, below
    arrow(ax, [(13.2, 19.0), (14.5, 19.0), (14.5, 5.0), (80.0, 5.0), (80.0, 26.9)], color="#b5532a")
    text(ax, 46.0, 6.2, "$x_B$ (32 numbers, transformed)", size=9.5, color="#b5532a")
    # outputs
    titled_panel(ax, 91.5, 8, 7.6, 43, "", BLUE)
    text(ax, 95.3, 49.0, "output $y$\n(64)", size=11, linespacing=1.2)
    column(ax, 95.3, [43.0, 40.5, 38.0, None, 33.0], None, fc="#cfe0f7")
    column(ax, 95.3, [24.0, 21.5, 19.0, None, 14.0], None, fc="#fbd9c2")
    text(ax, 95.3, 46.0, "$y_A = x_A$", size=9.5, color="#1f5fbf")
    text(ax, 95.3, 28.2, "$y_B$", size=9.5, color="#b5532a")
    arrow(ax, [(13.2, 45.5), (91.4, 45.5)], color="#1f5fbf", ls="--")
    text(ax, 40.0, 52.2, "", size=8)
    arrow(ax, [(89.6, 28.5), (91.4, 24.0)], color=FWD)
    # formulas
    panel(ax, 14.0, 0.4, 72.0, 4.3, fc="#fff8e7", r=0.8)
    text(ax, 50.0, 2.55, "forward:  $y_A = x_A,\\ \\ y_B = x_B\\odot e^{s(x_A)} + t(x_A)$        inverse (exact):  $x_A = y_A,\\ \\ x_B = (y_B - t(y_A))\\odot e^{-s(y_A)}$   (s, t are computed from the kept half, which is unchanged)", size=9.0)
    return fig


def inverse_figure():
    fig, ax = new_figure(19)
    text(ax, 50, 54.6, "Invertible DeepONet (Q64): inverse pass, target ERP $\\rightarrow$ several designs (no iteration, no VAE)", size=14)
    bw, bh = 21.0, 19.0
    xs = [3.0, 28.0, 53.0, 78.0]
    y1, y2 = 31.0, 6.0
    def stage(x, y, n, title, body, shape, fc):
        panel(ax, x, y, bw, bh, fc=fc, r=1.4, z=2)
        text(ax, x + 1.2, y + bh - 1.8, n, size=13, ha="left", color=INV)
        text(ax, x + bw / 2 + 1.0, y + bh - 1.8, title, size=12.5)
        text(ax, x + bw / 2, y + bh / 2 - 0.5, body, size=10.2, linespacing=1.32)
        text(ax, x + bw / 2, y + 2.0, shape, size=10.2, color="#1d5c1d")
    stage(xs[0], y1, "1", "Target ERP", "measured or desired spectrum\nnormalised with the\ntraining mean and std", "301 points", "#e3ecf8")
    spectrum_inset(fig, ax, xs[0] + 5.0, y1 + 3.6, 11.0, 4.4, CURVE, color=INV)
    stage(xs[1], y1, "2", "Projection onto the basis", "least squares, closed form:\n$b^* = \\Psi^\\top (y-\\psi_0)\\,/\\,F$\n($\\Psi^\\top\\Psi = F\\,I$, QR step)", "$b^*$: 64 nodes", "#fbe5d6")
    stage(xs[2], y1, "3", "Sampling", "$b_s = b^* + \\varepsilon_s$\n$\\varepsilon_s\\sim\\mathcal{N}(0,\\,\\sigma^2 I/F)$\n$\\sigma^2$: 0.068 (Q64), 0.133 (Q64-ERP)\nsample 0: no noise (point estimate)", "16 $\\times$ 64", "#fdf3c9")
    stage(xs[3], y1, "4", "Inverse of the RealNVP", "$T^{-1}$: the 10 couplings,\nin reverse order\n$x_B=(y_B-t)\\odot e^{-s}$\nexact: $T^{-1}(T(a))=a$", "16 $\\times$ [$a$ (8) $\\mid$ $z$ (56)]", "#fbe5d6")
    stage(xs[3], y2, "5", "Keep the design part", "take the first 8 numbers $a$\n$z$ (56 numbers) should be $\\approx 0$\n(trained with $0.1\\,\\mathrm{Huber}(z,0)$)", "16 $\\times$ 8", "#ececec")
    stage(xs[2], y2, "6", "Decode to physical units", "bounded logit $\\rightarrow$ sigmoid:\nm $\\in$ [0.1, 1] kg, $f_t\\in$ [10, 160] Hz\nx $\\in$ [0.05, 1.35], y $\\in$ [0.05, 0.45] m\n$k=m(2\\pi f_t)^2$", "16 $\\times$ 2 $\\times$ [m, k, $f_t$, x, y]", "#e2f1e2")
    stage(xs[1], y2, "7", "Sort and check", "sort resonators by $f_t$;\nsolve each design with the\nplate solver, compare with\nthe target ERP", "16 candidate designs + ERPs", "#fbe3e8")
    stage(xs[0], y2, "8", "Output", "point estimate (sample 0), or\nbest of 16 chosen with the solver\n(oracle: needs the target)", "design(s) $\\hat a$", "#e3ecf8")
    for i in range(3):
        arrow(ax, [(xs[i] + bw + 0.2, y1 + bh / 2), (xs[i + 1] - 0.2, y1 + bh / 2)], color=INV, lw=2.0)
    arrow(ax, [(xs[3] + bw / 2, y1 - 0.2), (xs[3] + bw / 2, y2 + bh + 0.2)], color=INV, lw=2.0)
    for i in (3, 2, 1):
        arrow(ax, [(xs[i] - 0.2, y2 + bh / 2), (xs[i - 1] + bw + 0.2, y2 + bh / 2)], color=INV, lw=2.0)
    text(ax, 50, 2.0, "Everything is one forward pass of the same trained network: the trunk basis $\\Psi$ and $\\psi_0$ come from the trunk, $T^{-1}$ from the same weights as $T$. "
                      "Several different designs come from the noise on $b$, not from a VAE.", size=10.2)
    return fig


def table_figure():
    fig, ax = new_figure(19)
    text(ax, 50, 54.6, "Invertible DeepONet: layers, sizes, parameters, training and results", size=14)
    # left: layer table
    panel(ax, 1, 1.5, 61, 51.0, fc="white", r=1.4)
    hdr = ["Module", "Layer / operation", "Size in $\\rightarrow$ out", "Parameters"]
    cx = [2.2, 15.5, 38.5, 53.0]
    for x, h in zip(cx, hdr):
        text(ax, x, 50.2, h, size=10, ha="left", color=INV)
    ax.plot([1.8, 61.2], [48.9, 48.9], color=EDGE, lw=0.8)
    rows = [
        ("Input", "design $a$ (bounded logit of m, $f_t$, x, y)", "8", "-"),
        ("", "append 56 zeros", "8 $\\rightarrow$ 64", "-"),
        ("Branch $T$", "coupling MLP: Linear + SiLU", "64 $\\rightarrow$ 256", "16,640"),
        ("10 couplings", "coupling MLP: Linear + SiLU", "256 $\\rightarrow$ 256", "65,792"),
        ("", "coupling MLP: Linear (log s, t)", "256 $\\rightarrow$ 128", "32,896"),
        ("", "affine map $y_B=x_B e^{s}+t$", "64 $\\rightarrow$ 64", "0"),
        ("", "one coupling layer / the 10 layers", "", "115,328 / 1,153,280"),
        ("Trunk", "Fourier features of $f$", "1 $\\rightarrow$ 65", "0 (fixed $k\\pi$)"),
        ("(per frequency)", "Linear + SiLU", "65 $\\rightarrow$ 256", "16,896"),
        ("", "3 $\\times$ [Linear + SiLU]", "256 $\\rightarrow$ 256", "3 $\\times$ 65,792"),
        ("", "Linear", "256 $\\rightarrow$ 65", "16,705"),
        ("", "trunk total", "", "230,977"),
        ("Basis", "split $\\psi_0$ (1) and $\\tilde\\Psi$ (64); QR: $\\Psi^\\top\\Psi=F I$", "301 $\\times$ 64", "0"),
        ("Combine", "$\\mathrm{ERP}=\\Psi\\,b+\\psi_0$", "(301 $\\times$ 64)(64) $\\rightarrow$ 301", "0"),
        ("Total Q64", "(Q8: Q = D = 8, no padding)", "", "1,384,257  (Q8: 938,665)"),
    ]
    y = 46.6
    for mod, op, io, pr in rows:
        bold = mod.startswith("Total") or op.endswith("total") or "one coupling" in op
        if bold:
            panel(ax, 1.8, y - 1.4, 59.4, 2.8, fc="#fdf3c9", ec="none", r=0.4, z=1)
        text(ax, cx[0], y, mod, size=9.2, ha="left", color="#1f5fbf")
        text(ax, cx[1], y, op, size=9.2, ha="left")
        text(ax, cx[2], y, io, size=9.2, ha="left")
        text(ax, cx[3], y, pr, size=9.2, ha="left")
        y -= 2.85
    text(ax, 2.2, 3.4, "Q64 checkpoints: erp$\\_$invertible$\\_$deeponet/models/200k$\\_$2res$\\_$18modes/idon$\\_$q64.pth, idon$\\_$q64-erp.pth (same sizes); Q8: idon$\\_$q8.pth", size=7.8, ha="left", color="#555555")
    # right panels
    def info(y0, h, title, lines, fc):
        panel(ax, 63.5, y0, 35.5, h, fc=fc, r=1.4)
        text(ax, 65.0, y0 + h - 1.6, title, size=10.5, ha="left", color=INV)
        for i, ln in enumerate(lines):
            text(ax, 65.0, y0 + h - 4.2 - 2.05 * i, ln, size=8.8, ha="left")
    info(36.5, 16.0, "Sizes", ["design $D=8$ (2 resonators $\\times$ 4), pad 56, $Q=64$", "$F=301$ frequencies (10-160 Hz, 0.5 Hz)",
                               "coupling: 2 hidden layers of 256, SiLU", "trunk: 4 hidden layers of 256, SiLU, 32 Fourier pairs",
                               "10 couplings; masks: random halves (Q64),", "even / odd alternating (Q8, $D\\leq 16$)"], "#e3ecf8")
    info(20.0, 15.5, "Training (150 epochs)", ["data: 200k two-resonator configs, 18 plate modes,", "split 160k / 20k / 20k (LHS)", "Adam, lr $5\\times10^{-4}$, cosine to $0.01\\,$lr, clip 5, batch 128",
                                                "loss = fwd + $w(e)$ Huber(inv) + 0.1 Huber($z$,0)", "$w(e)=\\min(1,(e+1)/10)$; last coupling layer zero-init",
                                                "Q64-ERP: fwd = MSE + 0.5 slope + 0.05 peak (from ep. 121)"], "#fdf3c9")
    info(1.5, 16.5, "Results (test; 500 targets, 16 samples)", ["Q8:  fwd RMSE 5.98 dB, inverse ERP 9.97 dB, 24.4 cm",
                                                              "Q64: fwd 2.58 dB, peak err 16.7 dB, inverse 6.66 dB, 9.9 cm",
                                                              "Q64-ERP: fwd 3.65 dB, peak err 7.7 dB, inverse 6.80 dB, 9.9 cm",
                                                              "(inverse = solver ERP of the point estimate vs target;",
                                                              "best-of-16 oracle: 7.97 / 5.07 / 5.50 dB; Q8 early-stopped at 50)",
                                                              "inverse $\\sim$0.2 ms per target (batched); no iteration"], "#e2f1e2")
    return fig


if __name__ == "__main__":
    for name, fn in (("idon_arch_forward", forward_figure), ("idon_arch_coupling", coupling_figure),
                     ("idon_arch_inverse", inverse_figure), ("idon_arch_table", table_figure)):
        save(fn(), name)
