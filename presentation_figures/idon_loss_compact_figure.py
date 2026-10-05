"""Invertible DeepONet loss as short equations with simple symbols and the meaning of every symbol: base loss (Q8, Q64) and ERP loss (Q64-ERP: adds slope and peak terms)."""
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
    "idon_base_loss_compact": [
        r"$\mathcal{L}=\mathcal{L}_{\mathrm{ERP}}+w\,\mathcal{L}_{\mathrm{design}}+0.1\,\mathcal{L}_{\mathrm{pad}}$",
        r"$\mathcal{L}_{\mathrm{ERP}}=\mathrm{mean}\,(\hat y-y)^2$",
        r"$\mathcal{L}_{\mathrm{design}}=\mathrm{Huber}(\hat a,\,a),\qquad \mathcal{L}_{\mathrm{pad}}=\mathrm{Huber}(\hat z,\,0)$",
        r"$w=\min\left(1,\ (e+1)/10\right)$",
        r"$a$: resonator design $(m,\ f_t,\ x,\ y)$ of each resonator;  $\hat a$: design recovered from the true ERP by the inverse",
        r"$z$: extra padded values of the design (zero going forward);  $\hat z$: their recovered values",
        r"$y$: true ERP;  $\hat y$: predicted ERP;  $e$: current epoch;  $E$: total number of epochs",
    ],
    "idon_erp_loss_compact": [
        r"$\mathcal{L}=\mathcal{L}_{\mathrm{ERP}}+0.5\,\mathcal{L}_{\mathrm{slope}}+\lambda_p\,\mathcal{L}_{\mathrm{peak}}+w\,\mathcal{L}_{\mathrm{design}}+0.1\,\mathcal{L}_{\mathrm{pad}}$",
        r"$\mathcal{L}_{\mathrm{ERP}}=\mathrm{mean}\,(\hat y-y)^2,\qquad \mathcal{L}_{\mathrm{slope}}=\mathrm{mean}\,(\Delta\hat y-\Delta y)^2$",
        r"$\mathcal{L}_{\mathrm{peak}}=\sum_{\mathrm{peaks}}(\hat y-y)^2,\qquad \lambda_p=0.05$ if $e>0.8\,E$, else $0$",
        r"$\mathcal{L}_{\mathrm{design}}=\mathrm{Huber}(\hat a,\,a),\qquad \mathcal{L}_{\mathrm{pad}}=\mathrm{Huber}(\hat z,\,0),\qquad w=\min\left(1,\ (e+1)/10\right)$",
        r"$\Delta$: difference between neighbouring frequencies;  peaks: resonance peaks of the true ERP",
        r"$a$: resonator design $(m,\ f_t,\ x,\ y)$ of each resonator;  $\hat a$: design recovered from the true ERP by the inverse",
        r"$z$: extra padded values of the design (zero going forward);  $\hat z$: their recovered values",
        r"$y$: true ERP;  $\hat y$: predicted ERP;  $e$: current epoch;  $E$: total number of epochs",
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
