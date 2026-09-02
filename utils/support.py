"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Utilities
Description : Generic utilities
==================================================
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import qmc


# ==================================================
# Device
# ==================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================================================
# Reproducibility
# ==================================================

def seed_everything(seed: int = 727) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")


# ==================================================
# Grid Generation
# ==================================================

def grid(Lx: float, Ly: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a uniform rectangular grid and return X, Y and cell area dA."""
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be >= 2.")

    x = np.linspace(0.0, Lx, nx, dtype=np.float64)
    y = np.linspace(0.0, Ly, ny, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    return X, Y, dx * dy


# ==================================================
# Latin Hypercube Sampling
# ==================================================

def lhs_sampling(
    num_samples: int,
    bounds: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Latin-hypercube samples scaled to ``bounds[:, 0:2]``."""
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_dimensions, 2).")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    sampler = qmc.LatinHypercube(d=bounds.shape[0], seed=seed)
    samples = sampler.random(n=num_samples)
    return qmc.scale(samples, bounds[:, 0], bounds[:, 1])


# ==================================================
# File-system helpers
# ==================================================

def _ensure_parent_dir(filename: str | os.PathLike[str]) -> Path:
    path = Path(filename)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ==================================================
# Dataset IO
# ==================================================

def save_dataset(dataset: Any, filename: str | os.PathLike[str]) -> None:
    path = _ensure_parent_dir(filename)
    torch.save(dataset, path)
    print(f"Dataset saved: {path}")


def load_dataset(filename: str | os.PathLike[str]) -> Any:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    dataset = torch.load(path, map_location="cpu", weights_only=False)
    print(f"Dataset loaded: {path}")
    return dataset


# ==================================================
# Model IO
# ==================================================

def save_model(model: torch.nn.Module, filename: str | os.PathLike[str]) -> None:
    path = _ensure_parent_dir(filename)
    torch.save(model.state_dict(), path)
    print(f"Model saved: {path}")


def load_model(
    model: torch.nn.Module,
    filename: str | os.PathLike[str],
    map_location: torch.device | str = device,
) -> torch.nn.Module:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()
    print(f"Model loaded: {path}")
    return model
