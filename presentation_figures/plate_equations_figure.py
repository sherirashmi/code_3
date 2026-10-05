"""Equations used by the solver, from the plate-resonator PDEs to the ERP.

Each step is read from utils/solver.py and utils/physics.py:
modal expansion (phi_mn, omega_mn), coupled matrices (compute_coupled_matrices), harmonic solve
(K + j w C - w^2 M) x = f (solve_modal_response), velocity v = j w w (compute_velocity), power and ERP (compute_erp,
compute_erp_spectrum, with the modal Gram matrix G).
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

TITLE, SUB, GRAY = "#1c3550", "#4a5a6a", "#6a7886"
STEPS = [
    ("1", "Plate with a point force and three sprung-mass resonators", [
        r"$D\nabla^4 w+\rho h\,\ddot w=f_d(t)\,\delta(x-x_d)\,\delta(y-y_d)-\sum_{i=1}^{3}\delta(x-x_i)\,\delta(y-y_i)\,\left[k_i(w-u_i)+c_i(\dot w-\dot u_i)\right]$",
        r"$m_i\ddot u_i+c_i(\dot u_i-\dot w_i)+k_i(u_i-w_i)=0,\qquad w_i=w(x_i,y_i,t),\qquad i=1,2,3$",
        r"$D=\dfrac{Eh^3}{12(1-\nu^2)},\qquad k_i=m_i\,(2\pi f_{t,i})^2$",
    ]),
    ("2", "Modal expansion of the plate (simply supported edges)", [
        r"$w(x,y,t)=\sum_{m,n}q_{mn}(t)\,\phi_{mn}(x,y),\qquad \phi_{mn}=\dfrac{2}{\sqrt{\rho h L_xL_y}}\,\sin\dfrac{m\pi x}{L_x}\,\sin\dfrac{n\pi y}{L_y}$",
        r"$\omega_{mn}=\sqrt{\dfrac{D}{\rho h}}\left[\left(\dfrac{m\pi}{L_x}\right)^2+\left(\dfrac{n\pi}{L_y}\right)^2\right],\qquad N$ modes ($15\times10$, or $6\times3$ for the 18-mode data)",
    ]),
    ("3", "Coupled modal system (plate modes and the three resonators)", [
        r"$\mathbf{M}\ddot{\mathbf{x}}+\mathbf{C}\dot{\mathbf{x}}+\mathbf{K}\mathbf{x}=\mathbf{f},\qquad \mathbf{x}=[q_1,\ldots,q_N,\,u_1,u_2,u_3]^{\top},\qquad \mathbf{M}=\mathrm{diag}(1,\ldots,1,\,m_1,m_2,m_3)$",
        r"$\mathbf{K}_{qq}=\mathrm{diag}(\omega_{mn}^2)+\sum_{i}k_i\,\vec\phi_i\vec\phi_i^{\top},\qquad \mathbf{K}_{qu_i}=-k_i\vec\phi_i,\qquad \mathbf{K}_{u_iu_i}=k_i,\qquad \vec\phi_i=[\phi_{mn}(x_i,y_i)]$",
        r"$\mathbf{C}$: the same with $c_i$ in place of $k_i$ and without the $\omega_{mn}^2$ term;$\qquad$ $\mathbf{f}=[F_0\,\phi_{mn}(x_d,y_d),\,0,0,0]^{\top}$",
    ]),
    ("4", r"Harmonic response ($e^{j\omega t}$, $\omega=2\pi f$)", [
        r"$\left(\mathbf{K}+j\omega\,\mathbf{C}-\omega^2\mathbf{M}\right)\hat{\mathbf{x}}(\omega)=\mathbf{f}\quad\Rightarrow\quad \hat q_{mn}(\omega),\ \hat u_i(\omega)$",
    ]),
    ("5", "Velocity, radiated power and ERP", [
        r"$\hat w(x,y,\omega)=\sum_{m,n}\hat q_{mn}\,\phi_{mn},\qquad \hat v=j\omega\,\hat w$",
        r"$P(f)=\dfrac{1}{2}\,\rho_Lc_L\iint|\hat v|^2\,dA=\dfrac{1}{2}\,\rho_Lc_L\,\omega^2\;\hat{\mathbf{q}}^{H}\mathbf{G}\,\hat{\mathbf{q}},\qquad G_{ij}=\iint\phi_i\phi_j\,dA$",
        r"$\mathrm{ERP}(f)=10\log_{10}\dfrac{P(f)}{P_{\mathrm{ref}}}$",
    ]),
]
LINE = {"1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0, "5": 1.0}

fig = plt.figure(figsize=(16, 14.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 148)
ax.axis("off")
ax.text(80, 144.0, "From the plate-resonator equations to the ERP", fontsize=22, ha="center", va="center", color=TITLE)

y = 139.0
for num, title, eqs in STEPS:
    h = 10.2 + 7.2 * (len(eqs) - 1) + 4.4
    ax.add_patch(FancyBboxPatch((3, y - h), 154, h, boxstyle="round,pad=0,rounding_size=1.6", fc="white", ec="#b8c8da", lw=1.4, zorder=1))
    ax.add_patch(plt.Circle((8.2, y - 3.6), 2.1, fc="#d9e8f7", ec="#8fb4d9", lw=1.4, zorder=2))
    ax.text(8.2, y - 3.6, num, fontsize=16, ha="center", va="center", color=TITLE, zorder=3)
    ax.text(12.5, y - 3.6, title, fontsize=15.5, ha="left", va="center", color=GRAY, zorder=3)
    yy = y - 10.4
    for eq in eqs:
        ax.text(80, yy, eq, fontsize=17.2 if len(eq) < 190 else 15.6, ha="center", va="center", color="#111111", zorder=3)
        yy -= 7.2
    y -= h + 1.6

ax.text(80, 4.5, r"$L_x=1.4$ m, $L_y=0.5$ m, $h=5$ mm, $E=71$ GPa, $\rho=2800\ \mathrm{kg/m^3}$, $\nu=0.3$, $F_0=1$ N at $(x_d,y_d)=(0.865,0.309)$ m;  "
                r"resonators: $m_i\in[0.1,1]$ kg, $f_{t,i}\in[10,160]$ Hz, $c_i=1$ N s/m;  $\rho_L=1.21\ \mathrm{kg/m^3}$, $c_L=343$ m/s, $P_{\mathrm{ref}}=10^{-12}$ W",
        fontsize=11.2, ha="center", va="center", color="#444444")
for ext, kw in (("png", dict(dpi=200)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"plate_equations.{ext}", facecolor="white", **kw)
print("saved")
