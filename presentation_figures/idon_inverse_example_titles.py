"""Inverse-design example of the Invertible DeepONet (Q64 and Q64-ERP): in the existing evaluation images
(erp_invertible_deeponet/plots/models/200k_2res_18modes/<variant>/inverse_example_3.png) the figure title ("... test configuration 3") is
cropped away and the title of the left plot is replaced by the loss equation of the model. Curves, positions and legends are the untouched
original pixels; the two panels are stacked.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import utils.plot_style  # noqa: F401

BASE = ROOT / "erp_invertible_deeponet" / "plots" / "models" / "200k_2res_18modes"
EXAMPLE = 3
TITLES = {
    "Q64": r"$\mathcal{L}=\mathcal{L}_{\mathrm{MSE}}+w(e)\,\mathcal{L}_{\mathrm{inv}}+0.1\,\mathcal{L}_{\mathrm{lat}}$",
    "Q64-ERP": r"$\mathcal{L}=\mathcal{L}_{\mathrm{MSE}}+0.5\,\mathcal{L}_{\mathrm{slope}}+\lambda_p(e)\,\mathcal{L}_{\mathrm{peak}}"
               r"+w(e)\,\mathcal{L}_{\mathrm{inv}}+0.1\,\mathcal{L}_{\mathrm{lat}}$",
}
DPI = 295  # of the evaluation images (15 x 5.2 in -> 4427 x 1509 px)
OUT_WIDTH = 3000
TITLE_SIZE = 14


def retitled(variant: str) -> Image.Image:
    img = Image.open(BASE / variant / f"inverse_example_{EXAMPLE}.png").convert("RGB")
    ink = np.asarray(img).min(axis=2) < 245
    W = img.width
    rows = np.where(ink.any(axis=1))[0]
    r = int(rows[0])
    while ink[r + 1].any():  # end of the old figure title ("Invertible DeepONet ...: inverse design of test configuration 3")
        r += 1
    nxt = r + 1
    while not ink[nxt].any():  # start of the next element: the axes titles
        nxt += 1
    frame = nxt
    while ink[frame, : W // 2].mean() < 0.4:  # top edge of the left axes
        frame += 1
    old = ink[nxt : frame - 3, : W // 2]  # old left title "ERP of the designs (checked with the solver)"
    cols = np.where(old.any(axis=0))[0]
    cmin, cmax = int(cols.min()), int(cols.max())
    h = frame - 3 - nxt
    out = img.copy()
    white = Image.new("RGB", (cmax - cmin + 24, h), "white")
    out.paste(white, (cmin - 12, nxt))
    fig = plt.figure(figsize=(white.width * 1.6 / DPI, h / DPI), dpi=DPI)
    fig.text(0.5, 0.5, TITLES[variant], fontsize=TITLE_SIZE, ha="center", va="center")
    fig.canvas.draw()
    strip = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    plt.close(fig)
    cx = (cmin + cmax) // 2
    out.paste(strip, (cx - strip.width // 2, nxt))
    return out.crop((0, nxt - 30, W, img.height))  # the old figure title is cropped away


panels = [retitled(v) for v in TITLES]
panels = [p.resize((OUT_WIDTH, round(p.height * OUT_WIDTH / p.width)), Image.LANCZOS) for p in panels]
canvas = Image.new("RGB", (OUT_WIDTH, sum(p.height for p in panels)), "white")
y = 0
for p in panels:
    canvas.paste(p, (0, y))
    y += p.height
canvas.save(ROOT / "presentation_figures" / "idon_inverse_q64_vs_q64erp.png", optimize=True)
print("saved", canvas.size)
