"""Collage of the inverse examples: one image per test configuration with the
plots of every variant stacked (same target ERP, different methods).

Reads plots/<dataset>/<variant>/inverse_example_<k>.png written by evaluate.py
and saves plots/<dataset>/ALL_MODELS/inverse_example_<k>_all_variants.png.

Usage:  python -m erp_invertible_deeponet.collage_examples [Q8 Q64 Q128 Q64-FNO Q64-DCO]
"""

from __future__ import annotations

import sys

from PIL import Image

from .train import DATASET, VARIANTS, plot_dir

WIDTH = 3000  # px of each stacked panel


def collage(names, k: int) -> None:
    panels = []
    for name in names:
        path = plot_dir(DATASET, name) / f"inverse_example_{k}.png"
        if path.exists():
            img = Image.open(path).convert("RGB")
            panels.append(img.resize((WIDTH, round(img.height * WIDTH / img.width)), Image.LANCZOS))
    if not panels:
        return
    out = Image.new("RGB", (WIDTH, sum(p.height for p in panels)), "white")
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height
    path = plot_dir(DATASET) / f"inverse_example_{k}_all_variants.png"
    out.save(path, optimize=True)
    print(f"Saved {path} ({len(panels)} variants)")


if __name__ == "__main__":
    names = sys.argv[1:] or ["Q8", "Q64", "Q128", "Q64-FNO", "Q64-DCO"]
    names = [n for n in names if n in VARIANTS]
    for k in (1, 2, 3):
        collage(names, k)
