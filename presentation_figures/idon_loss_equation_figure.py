"""Training loss of the Invertible DeepONet with the ERP loss (Q64-ERP): erp_invertible_deeponet/scripts/train.py and model.py.
MSE + 0.5 slope + peak term from 80 % of the epochs + inverse and latent Huber terms. Same style as loss_function."""
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
    "idon_loss_function": [
        r"$\mathcal{L}_e=\mathcal{L}_{\mathrm{MSE}}+\lambda_s\,\mathcal{L}_{\mathrm{slope}}+\lambda_p(e)\,\mathcal{L}_{\mathrm{peak}}+w(e)\,\mathcal{L}_{\mathrm{inv}}+0.1\,\mathcal{L}_{\mathrm{lat}},\qquad \lambda_s=0.5$",
        r"$\lambda_p(e)=0.05\ \ \mathrm{for}\ \ e\geq 0.8\,E,\ \ \mathrm{else}\ \ 0,\qquad w(e)=\min\!\left(1,\dfrac{e+1}{10}\right)$",
        r"$\mathcal{L}_{\mathrm{MSE}},\ \mathcal{L}_{\mathrm{slope}},\ \mathcal{L}_{\mathrm{peak}}$ on the ERP $\hat y=\Psi\,\mathbf{b}(\mathbf{a})+\psi_0$, as in the forward loss",
        r"$\mathcal{L}_{\mathrm{inv}}=\mathrm{Huber}(\hat{\mathbf{a}},\mathbf{a}),\qquad (\hat{\mathbf{a}},\hat{\mathbf{z}})=T^{-1}(\mathbf{b}^{*}),\qquad \mathbf{b}^{*}=\dfrac{1}{F}\,\Psi^{\top}(y-\psi_0)$",
        r"$\mathcal{L}_{\mathrm{lat}}=\mathrm{Huber}(\hat{\mathbf{z}},\mathbf{0})$",
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
