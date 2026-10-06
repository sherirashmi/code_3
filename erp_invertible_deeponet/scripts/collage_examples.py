"""Collage of the inverse examples: one image per test configuration with the
plots of every variant stacked (same target ERP, different methods).

Reads plots/<dataset>/<variant>/inverse_example_<k>.png written by evaluate.py
and saves plots/<dataset>/ALL_MODELS/inverse_example_<k>_all_variants.png.

Usage:  python -m erp_invertible_deeponet.scripts.collage_examples [Q8 Q64 Q128 Q64-FNO Q64-DCO]
"""

from __future__ import annotations

import sys

from PIL import Image

from .train import DATASET, VARIANTS, plot_dir

WIDTH = 3000  # px of each stacked panel


def collage(names, k: int, dataset_tag: str = DATASET) -> None:
    panels = []
    for name in names:
        path = plot_dir(dataset_tag, name) / f"inverse_example_{k}.png"
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
    path = plot_dir(dataset_tag) / f"inverse_example_{k}_all_variants.png"
    out.save(path, optimize=True)
    print(f"Saved {path} ({len(panels)} variants)")


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = DATASET
    if "--dataset" in args:
        i = args.index("--dataset")
        tag = args[i + 1]
        args = args[:i] + args[i + 2:]
    names = [n for n in (args or list(VARIANTS)) if n in VARIANTS]
    for k in (1, 2, 3):
        collage(names, k, tag)
