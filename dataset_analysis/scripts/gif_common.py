"""Shared helpers for the dataset-analysis GIFs: thesis (Computer Modern) typography,
presentation-friendly labels, and the enhanced plate-displacement heatmap.

Run any script from the repository root, e.g.
    python dataset_analysis/scripts/make_position_sweep_videos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import utils.plot_style  # noqa: F401  (applies the thesis LaTeX / Computer Modern look)
from utils.physics import Lx, Ly, xf, yf



def video_dir(group: str) -> Path:
    """dataset_analysis/<group>/videos, where <group> is a dataset tag or ``solver_demos``."""
    path = ROOT / "dataset_analysis" / group / "videos"
    path.mkdir(parents=True, exist_ok=True)
    return path


OUT_DIR = video_dir("solver_demos")  # demonstrations of the plate solver (no dataset)
GIF_DPI = 100

# One fixed resonator mass for the single-resonator sweeps; stiffness follows from
# the tuning frequency, k = m (2 pi f_t)^2, exactly as in the dataset.
M_RES = 0.5

FREQ_LABEL = "Frequency (Hz)"
ERP_LABEL = "Equivalent radiated power, ERP (dB)"
DB_RANGE = 30.0  # dynamic range of the displacement heatmap in dB

ERP_COLOR = "#2a78d6"
TUNING_COLOR = "#c2412c"
MODE_COLOR = "#9a9a94"


def configuration_row(tuning_frequency: float, x: float, y: float, mass: float = M_RES) -> list[float]:
    """One resonator as [m, k, f_t, x, y] with k = m (2 pi f_t)^2."""
    return [mass, mass * (2.0 * np.pi * tuning_frequency) ** 2, tuning_frequency, x, y]


def position_text(x: float, y: float) -> str:
    return f"position at ({x:.2f} m, {y:.2f} m)"


def style_axes(ax):
    ax.grid(True, color="#d9d9d4", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def displacement_db(field: np.ndarray, reference: float) -> np.ndarray:
    """Displacement magnitude in dB relative to ``reference``, floored at -DB_RANGE."""
    ratio = np.maximum(np.abs(field) / reference, 10 ** (-DB_RANGE / 20.0))
    return 20.0 * np.log10(ratio)


class PlateHeatmap:
    """Plate displacement heatmap drawn to scale (equal aspect) with a dB colour scale,
    contour lines, the plate outline and the force / resonator positions."""

    LEVELS = (-24.0, -18.0, -12.0, -6.0)

    def __init__(self, ax, cax, first_field_db: np.ndarray, force_label="Excitation force",
                 resonator_label="Resonator"):
        self.ax = ax
        extent = [0.0, Lx, 0.0, Ly]
        self.extent = extent
        self.im = ax.imshow(first_field_db, extent=extent, origin="lower", aspect="equal",
                            cmap="magma", vmin=-DB_RANGE, vmax=0.0, interpolation="bicubic")
        ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, ec="black", lw=1.2))
        ax.plot([xf], [yf], marker="*", color="#35d0ff", ms=15, mec="black", mew=0.9, ls="",
                label=force_label, zorder=6)
        (self.res_marker,) = ax.plot([], [], marker="o", color="#7CFC00", ms=10, mec="black", mew=1.0,
                                     ls="", label=resonator_label, zorder=7)
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_xticks(np.arange(0, Lx + 1e-9, 0.2))
        ax.set_yticks(np.arange(0, Ly + 1e-9, 0.1))
        ax.set_xlabel("Position $x$ (m)")
        ax.set_ylabel("Position $y$ (m)")
        ax.tick_params(direction="out")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False)
        cbar = ax.figure.colorbar(self.im, cax=cax, orientation="horizontal")
        cbar.set_label("Displacement magnitude (dB relative to the maximum of each frame)")
        cbar.set_ticks(np.arange(-DB_RANGE, 1, 10))
        self._contours = None
        self.update(first_field_db, [])

    def update(self, field_db: np.ndarray, resonator_xy):
        self.im.set_data(field_db)
        if self._contours is not None:
            self._contours.remove()
        self._contours = self.ax.contour(
            np.linspace(0, Lx, field_db.shape[1]), np.linspace(0, Ly, field_db.shape[0]),
            field_db, levels=self.LEVELS, colors="white", linewidths=0.6, alpha=0.55)
        xs = [p[0] for p in resonator_xy]
        ys = [p[1] for p in resonator_xy]
        self.res_marker.set_data(xs, ys)


def make_two_panel_figure(width=14.0, height=5.2):
    """Left: ERP spectrum. Right: plate heatmap (to scale) with a horizontal colour bar."""
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[1.0, 0.07],
                          left=0.07, right=0.97, top=0.86, bottom=0.14, wspace=0.18, hspace=0.9)
    ax_erp = fig.add_subplot(gs[:, 0])
    ax_field = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, 1])
    ax_field.set_anchor("C")
    style_axes(ax_erp)
    return fig, ax_erp, ax_field, cax


def finish(fig, title: str):
    fig.suptitle(title, y=0.965, fontsize=13)
