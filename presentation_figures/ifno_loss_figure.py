"""Training loss of the invertible coupling-flow operators (iFNO and variants): erp_invertible/scripts/common.py (stage1_loss, stage2_loss, stage3_loss)
and train.py. Simple symbols, with the meaning of each term."""
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
    "ifno_loss_function": [
        r"Stage 3 (all parts together):  $\mathcal{L}=\mathcal{L}_{\mathrm{fwd}}+\mathcal{L}_{\mathrm{inv}}+\mathcal{L}_{\mathrm{dir},a}+\mathcal{L}_{\mathrm{dir},y}+\mathcal{L}_{\mathrm{VAE}}$",
        r"Stage 1 (invertible blocks only):  $\mathcal{L}=\mathcal{L}_{\mathrm{fwd}}+\mathcal{L}_{\mathrm{inv}}+\mathcal{L}_{\mathrm{dir},a}+\mathcal{L}_{\mathrm{dir},y}$",
        r"Stage 2 (VAE only):  $\mathcal{L}=\mathcal{L}_{\mathrm{VAE}}=\mathcal{L}_{\mathrm{recon}}+\beta\,\mathcal{L}_{\mathrm{KL}},\qquad \beta=0.05$",
        r"$\mathcal{L}_{\mathrm{fwd}}=\mathrm{mean}\,(\hat y-y)^2,\qquad \mathcal{L}_{\mathrm{inv}}=\mathrm{mean}\,(\hat a-a)^2$",
        r"$\mathcal{L}_{\mathrm{dir},y}=\mathrm{mean}\,(\tilde y-y)^2,\qquad \mathcal{L}_{\mathrm{dir},a}=\mathrm{mean}\,(\tilde a-a)^2,\qquad \mathcal{L}_{\mathrm{recon}}=\mathrm{mean}\,(a_{\mathrm{rec}}-a)^2$",
        r"$a$: design;  $y$: ERP;  $\mathcal{L}_{\mathrm{KL}}$: distance of the VAE latent distribution from $\mathcal{N}(0,1)$",
        r"$\hat y$, $\hat a$: ERP and design predicted through the invertible blocks (forward and inverse)",
        r"$\tilde y$, $\tilde a$: ERP and design read out directly from the lifted ERP and design (blocks skipped)",
        r"$a_{\mathrm{rec}}$: design reconstructed by the VAE",
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
