"""The equations from the plate-resonator PDEs to the ERP, one image per step (same font size in all of them).

Each step is read from utils/solver.py and utils/physics.py: modal expansion (phi_mn, omega_mn), coupled matrices
(compute_coupled_matrices), harmonic solve (K + j w C - w^2 M) x = f (solve_modal_response), velocity v = j w w
(compute_velocity), power and ERP (compute_erp).
"""
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
    "step1_governing_equations": [
        r"$D\nabla^4 w+\rho h\,\ddot w=f_d(t)\,\delta(x-x_d)\,\delta(y-y_d)-\sum_{i=1}^{3}\delta(x-x_i)\,\delta(y-y_i)\,\left[k_i(w-u_i)+c_i(\dot w-\dot u_i)\right]$",
        r"$m_i\ddot u_i+c_i(\dot u_i-\dot w_i)+k_i(u_i-w_i)=0,\qquad w_i=w(x_i,y_i,t),\qquad i=1,2,3$",
    ],
    "step2_mode_shapes_and_frequencies": [
        r"$\phi_{mn}=\dfrac{2}{\sqrt{\rho h L_xL_y}}\,\sin\dfrac{m\pi x}{L_x}\,\sin\dfrac{n\pi y}{L_y}$",
        r"$\omega_{mn}=\sqrt{\dfrac{D}{\rho h}}\left[\left(\dfrac{m\pi}{L_x}\right)^2+\left(\dfrac{n\pi}{L_y}\right)^2\right],\qquad N$ modes",
    ],
    "step3_coupled_modal_system": [
        r"$\mathbf{M}\ddot{\mathbf{x}}+\mathbf{C}\dot{\mathbf{x}}+\mathbf{K}\mathbf{x}=\mathbf{f},\qquad \mathbf{x}=[q_1,\ldots,q_N,\,u_1,u_2,u_3]^{\top},\qquad \mathbf{M}=\mathrm{diag}(1,\ldots,1,\,m_1,m_2,m_3)$",
        r"$\mathbf{K}_{qq}=\mathrm{diag}(\omega_{mn}^2)+\sum_{i}k_i\,\vec\phi_i\vec\phi_i^{\top},\qquad \mathbf{K}_{qu_i}=-k_i\vec\phi_i,\qquad \mathbf{K}_{u_iu_i}=k_i,\qquad \vec\phi_i=[\phi_{mn}(x_i,y_i)]$",
        r"$\mathbf{C}$: the same with $c_i$ in place of $k_i$ and without the $\omega_{mn}^2$ term;$\qquad$ $\mathbf{f}=[F_0\,\phi_{mn}(x_d,y_d),\,0,0,0]^{\top}$",
    ],
    "step4_harmonic_response": [
        r"$\left(\mathbf{K}+j\omega\,\mathbf{C}-\omega^2\mathbf{M}\right)\hat{\mathbf{x}}(\omega)=\mathbf{f}$",
    ],
    "step5_velocity_power_erp": [
        r"$\hat w(x,y,\omega)=\sum_{m,n}\hat q_{mn}\,\phi_{mn},\qquad \hat v=j\omega\,\hat w$",
        r"$P=\dfrac{1}{2}\,\rho_L\,c_L\,|\hat v|^2$",
        r"$\mathrm{ERP}(f)=10\log_{10}\dfrac{P(f)}{P_{\mathrm{ref}}}$",
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
        fig.savefig(ROOT / "presentation_figures" / f"plate_equation_{name}.{ext}", facecolor="white", **kw)
    plt.close(fig)
    print("saved", name, f"{W:.0f}x{H:.0f} pt")
