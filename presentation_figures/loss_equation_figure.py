"""Training loss of the forward ERP operators (erp_spectrum_loss in erp_forward/scripts/neural_operator_utils.py):
MSE + 0.5 * slope MSE + 0.05 * peak squared error, on normalised ERP spectra. Same style as the equation steps."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import utils.plot_style  # noqa: F401

FS = 17.0  # one font size for every equation in every image
PAD_X, PAD_Y, GAP = 26.0, 18.0, 16.0  # points
STEPS = {
    "loss_function": [
        r"$\mathcal{L}=\mathcal{L}_{\mathrm{MSE}}+\lambda_s\,\mathcal{L}_{\mathrm{slope}}+\lambda_p\,\mathcal{L}_{\mathrm{peak}},\qquad \lambda_s=0.5,\quad \lambda_p=0.05$",
        r"$\mathcal{L}_{\mathrm{MSE}}=\dfrac{1}{BF}\sum_{b=1}^{B}\sum_{f=1}^{F}\left(\hat y_{b,f}-y_{b,f}\right)^2$",
        r"$\mathcal{L}_{\mathrm{slope}}=\dfrac{1}{B(F-1)}\sum_{b=1}^{B}\sum_{f=1}^{F-1}\left(\Delta\hat y_{b,f}-\Delta y_{b,f}\right)^2,\qquad \Delta y_{b,f}=y_{b,f+1}-y_{b,f}$",
        r"$\mathcal{L}_{\mathrm{peak}}=\dfrac{1}{B}\sum_{b=1}^{B}\;\sum_{f\in\mathcal{P}(y_b)}\left(\hat y_{b,f}-y_{b,f}\right)^2,\qquad \mathcal{P}(y_b)=$ local maxima of the true spectrum",
    ],
}

for name, eqs in STEPS.items():
    # measure every line on a scratch figure
    fig = plt.figure(figsize=(30, 6))
    r = fig.canvas.get_renderer()
    sizes = []
    for eq in eqs:
        t = fig.text(0.5, 0.5, eq, fontsize=FS)
        bb = t.get_window_extent(r)
        sizes.append((bb.width * 72 / fig.dpi, bb.height * 72 / fig.dpi))
    plt.close(fig)
    W = max(w for w, _ in sizes) + 2 * PAD_X
    H = sum(h for _, h in sizes) + GAP * (len(eqs) - 1) + 2 * PAD_Y
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1.5, 1.5), W - 3, H - 3, boxstyle="round,pad=0,rounding_size=10", fc="white", ec="#b8c8da", lw=1.4))
    y = H - PAD_Y
    for eq, (_, h) in zip(eqs, sizes):
        ax.text(W / 2, y - h / 2, eq, fontsize=FS, ha="center", va="center", color="#111111")
        y -= h + GAP
    for ext, kw in (("png", dict(dpi=250)), ("pdf", {}), ("svg", {})):
        fig.savefig(ROOT / "presentation_figures" / f"{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", name, f"{W:.0f}x{H:.0f} pt")
