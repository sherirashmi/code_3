"""Shared helpers for the true-vs-predicted case studies (inference only, no training): case definitions, novelty check against the
training set, plate drawing and error metrics."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from matplotlib.patches import Rectangle

from erp_inverse.scripts.common import canonicalize_by_ft
from erp_forward.scripts.neural_operator_utils import _configuration_features, _split_ids, spectrum_peak_mask
from utils.physics import Lx, Ly, edge_margin, fmax, fmin, m_max, m_min, xf, yf

OUT = Path(__file__).resolve().parent
RES_COLORS = ("#2a78d6", "#eb6834", "#1f9e74")


def build_configuration(rows) -> np.ndarray:
    """(m [kg], f_t [Hz], x [m], y [m]) rows -> (R, 5) [m, k, f_t, x, y] with k = m (2 pi f_t)^2."""
    return np.asarray([[m, m * (2 * np.pi * f) ** 2, f, x, y] for m, f, x, y in rows], dtype=np.float32)


def out_of_range(design) -> list[str]:
    notes = []
    for m, _, f_t, x, y in design:
        if not m_min <= m <= m_max:
            notes.append(f"m outside {m_min:g}-{m_max:g} kg")
        if not fmin <= f_t <= fmax:
            notes.append(f"tuning frequency outside {fmin:g}-{fmax:g} Hz")
        if not (edge_margin <= x <= Lx - edge_margin and edge_margin <= y <= Ly - edge_margin):
            notes.append("closer to the edge than 5 cm")
    return sorted(set(notes))


def _vector(physical: np.ndarray) -> np.ndarray:
    """Design -> range-scaled vector of (m, f_t, x, y) of the resonators sorted by f_t (every number in [0, 1] inside the training ranges)."""
    p = canonicalize_by_ft(np.asarray(physical, dtype=np.float64))
    lo = np.array([m_min, fmin, edge_margin, edge_margin])
    span = np.array([m_max - m_min, fmax - fmin, Lx - 2 * edge_margin, Ly - 2 * edge_margin])
    return ((p[..., [0, 2, 3, 4]] - lo) / span).reshape(*p.shape[:-2], -1)


def training_distance(dataset, designs: np.ndarray, n_reference: int = 300) -> tuple[np.ndarray, float]:
    """Distance (RMS over the numbers of the range-scaled design) of each design to its nearest TRAINING configuration, and the median of the
    same distance for held-out test configurations (reference for 'how unusual is it')."""
    ids = _split_ids(dataset)
    feats = _configuration_features(dataset).astype(np.float64)
    train = torch.from_numpy(_vector(feats[ids["train"]]))
    scale = np.sqrt(train.shape[1])

    def nearest(vecs, exclude_self=False):
        d = torch.cdist(torch.from_numpy(np.atleast_2d(vecs)), train)
        return d.min(dim=1).values.numpy() / scale

    mine = nearest(_vector(np.asarray(designs, dtype=np.float64)))
    reference = float(np.median(nearest(_vector(feats[ids["test"][:n_reference]]))))
    return mine, reference


def rmse(pred, true) -> float:
    return float(np.sqrt(((np.asarray(pred) - np.asarray(true)) ** 2).mean()))


def peak_rmse(pred, true) -> float:
    mask = spectrum_peak_mask(torch.from_numpy(np.asarray(true, dtype=np.float32))[None, :, None])[0, :, 0].numpy().astype(bool)
    return float(np.sqrt(((np.asarray(pred)[mask] - np.asarray(true)[mask]) ** 2).mean()))


def draw_plate(ax, designs: dict, marker_size=10):
    """designs: label -> (R, 5) design; the first entry is the truth (circles), the others crosses."""
    ax.add_patch(Rectangle((0, 0), Lx, Ly, facecolor="#f4f4f1", ec="black", lw=1.2, zorder=0))
    ax.add_patch(Rectangle((edge_margin, edge_margin), Lx - 2 * edge_margin, Ly - 2 * edge_margin, fill=False, ec="#9a9a94", ls="--", lw=0.8))
    ax.plot([xf], [yf], marker="*", color="#35d0ff", ms=14, mec="black", mew=0.8, ls="", zorder=6, label="Excitation force")
    for r, (m, _, f_t, x, y) in enumerate(designs["True"]):
        ax.plot(x, y, "o", color=RES_COLORS[r % 3], ms=marker_size + 1, mec="black", mew=1.0, ls="", zorder=5,
                label=f"R{r + 1}: m = {m:g} kg, f$_t$ = {f_t:g} Hz")
    ax.set_xlim(-0.04, Lx + 0.04)
    ax.set_ylim(-0.04, Ly + 0.04)
    ax.set_aspect("equal")
    ax.set_xlabel("Position x (m)")
    ax.set_ylabel("Position y (m)")
