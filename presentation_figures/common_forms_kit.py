"""Shared pieces of the common-form figures: variant cards, pipeline elements, loss panel."""
from diagram_kit import *  # noqa: F401,F403
from diagram_kit import (BLUE, GREEN, ORANGE, YELLOW, GRAY, PURPLE, PINK, EDGE, panel, text, box, arrow, circle_op,
                         diamond, spectrum_inset, sample_erp, new_figure, save)

FWD_COL, INV_COL = "#2a78d6", "#e8632b"
VARY_FILL, VARY_EDGE = "#fff1e3", "#e8632b"


def frame(ax, left_w, title, subtitle):
    """Two dashed outer panels (network | loss); returns the x range of the right panel."""
    panel(ax, 1.0, 4.4, left_w, 49.6, fc="#f6f6f6", ls="--", lw=1.2, r=2.0, z=0)
    rx = 1.0 + left_w + 1.4
    panel(ax, rx, 4.4, 99.2 - rx, 49.6, fc="#f6f6f6", ls="--", lw=1.2, r=2.0, z=0)
    text(ax, 1.0 + left_w / 2, 52.4, title, size=12.5)
    text(ax, 1.0 + left_w / 2, 50.6, subtitle, size=8.3, color="#555555")
    return rx


def elem(ax, x0, x1, y0, y1, title, body, fc, size=6.9, tsize=8.4, params=None, vary=False, ls="-"):
    if vary:
        panel(ax, x0, y0, x1 - x0, y1 - y0, fc=VARY_FILL, ec=VARY_EDGE, ls="--", lw=1.7, r=1.3, z=2)
    else:
        panel(ax, x0, y0, x1 - x0, y1 - y0, fc=fc, r=1.3, z=2, ls=ls)
    text(ax, (x0 + x1) / 2, y1 - 1.25, title, size=tsize)
    text(ax, (x0 + x1) / 2, (y0 + y1) / 2 - 0.5, body, size=size, linespacing=1.3, color="#222222")
    if params:
        text(ax, (x0 + x1) / 2, y0 + 0.9, params, size=6.3, color="#555555")


def card(ax, x, y, w, h, title, lines, params, size=7.5, tsize=8.6, tag=None):
    """Variant card: title, left-aligned text lines, parameter count at the bottom."""
    panel(ax, x, y, w, h, fc=VARY_FILL, ec=VARY_EDGE, ls="-", lw=1.1, r=0.9, z=2)
    text(ax, x + w / 2, y + h - 1.1, title, size=tsize)
    ax.text(x + 0.6, y + h - 2.7, "\n".join(lines), fontsize=size, ha="left", va="top", zorder=10, linespacing=1.4,
            color="#222222")
    text(ax, x + w / 2, y + 0.85, params, size=6.8, color="#555555")


def loss_panel(ax, rx, boxes, footer, data_label="Training data", box_h=6.2, top=45.4, footer_size=7.8):
    """Right-hand panel: training-data box, stacked loss boxes with a data bus, footer box."""
    cx = (rx + 99.2) / 2
    bx0, bx1 = rx + 1.4, 99.2 - 2.8
    text(ax, cx, 52.4, "Loss functions", size=12.5)
    box(ax, rx + 2.0, top, 99.2 - rx - 5.2 - 0.0, 4.2, data_label, fc=YELLOW, size=8.0)
    y = top - 1.9 - box_h
    mids = []
    for name, formula in boxes:
        panel(ax, bx0, y, bx1 - bx0, box_h, fc=GRAY, r=0.9, z=3)
        text(ax, (bx0 + bx1) / 2, y + box_h - 1.6, name, size=8.2)
        text(ax, (bx0 + bx1) / 2, y + 1.9, formula, size=7.4)
        mids.append(y + box_h / 2)
        y -= box_h + 1.0
    bus = 99.2 - 1.3
    ax.plot([rx + 2.0 + (99.2 - rx - 5.2), bus, bus], [top + 2.1, top + 2.1, mids[-1]], lw=1.0, color="black", zorder=6)
    for m in mids:
        arrow(ax, [(bus, m), (bx1 + 0.2, m)], lw=0.9, ms=6)
    fy1 = y + box_h + 1.0 - 1.0
    box(ax, bx0, 5.2, bx1 - bx0, fy1 - 5.2 - 0.6, footer, fc="#f7f0c8", size=footer_size)
