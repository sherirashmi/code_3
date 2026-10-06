"""Short form of the Kirchhoff-Love plate equation, bare (no box), like the Helmholtz equation image on the 'Different approaches' slide."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils.plot_style  # noqa: F401

EQ = r"$D\,\nabla^{4}w+\rho h\,\ddot{w}=q$"
fig = plt.figure(figsize=(5.2, 1.0))
fig.text(0.5, 0.5, EQ, fontsize=34, ha="center", va="center")
for ext, kw in (("png", dict(dpi=300)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"plate_equation_short.{ext}", facecolor="white", **kw)
fig.savefig(ROOT / "presentation_figures" / "plate_equation_short_transparent.png", dpi=300, transparent=True)
print("saved")
