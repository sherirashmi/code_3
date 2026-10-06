"""Animated explanation of the bounded design space (sigmoid decoding vs hard clipping).

The network predicts an unbounded number w; the physical design value (here the
x position on the 1.4 m plate) is recovered either with the sigmoid decoding used
in erp_inverse/scripts/design_space.py,

    x = lo + (hi - lo) * sigmoid(w),

or with a naive hard clip of a linear map with the same slope at the centre,

    x = clip(c + s * w, lo, hi).

Panels: (a) decoding curve, (b) gradient dx/dw (the learning signal),
(c) decoded values of a batch of predictions whose mean drifts to the plate edge.

Usage (from the repository root):
    python presentation_figures/bounded_design_animation.py
writes presentation_figures/bounded_design_animation.gif (+ a static .png of
the last frame).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.plot_style import apply_thesis_style  # noqa: E402

LO, HI = 0.0, 1.4  # plate length Lx (m): bounds of the x position
SLOPE = (HI - LO) / 4.0  # sigmoid slope at w = 0, so both maps agree in the centre
CENTRE = 0.5 * (LO + HI)
W = np.linspace(-6.0, 6.0, 600)
SIGMOID_C, CLIP_C = "#1f77b4", "#d62728"


def decode_sigmoid(w):
    return LO + (HI - LO) / (1.0 + np.exp(-w))


def decode_clip(w):
    return np.clip(CENTRE + SLOPE * w, LO, HI)


def grad_sigmoid(w):
    s = 1.0 / (1.0 + np.exp(-w))
    return (HI - LO) * s * (1.0 - s)


def grad_clip(w):
    raw = CENTRE + SLOPE * w
    return np.where((raw > LO) & (raw < HI), SLOPE, 0.0)


def main(out=ROOT / "presentation_figures" / "bounded_design_animation.gif", n_frames: int = 90, fps: int = 15):
    apply_thesis_style(usetex=False)
    rng = np.random.default_rng(727)
    noise = rng.normal(0.0, 1.0, 4000)  # batch of predictions around the moving mean
    # the mean output drifts from the centre beyond the right edge and back
    path = np.concatenate((np.linspace(0.0, 5.5, n_frames // 2), np.linspace(5.5, 0.0, n_frames - n_frames // 2)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax_map, ax_grad, ax_hist = axes

    # (a) decoding curve
    ax_map.axhspan(LO, HI, color="0.92", zorder=0, label="Physical range (plate)")
    ax_map.plot(W, decode_sigmoid(W), color=SIGMOID_C, lw=2.2, label="Sigmoid decoding")
    ax_map.plot(W, decode_clip(W), color=CLIP_C, lw=2.2, ls="--", label="Hard clipping")
    for b in (LO, HI):
        ax_map.axhline(b, color="0.4", lw=0.8, ls=":")
    pt_sig, = ax_map.plot([], [], "o", color=SIGMOID_C, ms=9)
    pt_clip, = ax_map.plot([], [], "s", color=CLIP_C, ms=8)
    ax_map.set_xlim(W[0], W[-1])
    ax_map.set_ylim(LO - 0.25, HI + 0.25)
    ax_map.set_xlabel(r"Network output $w$")
    ax_map.set_ylabel(r"Decoded position $x$ (m)")
    ax_map.set_title("(a) Decoding")
    ax_map.legend(loc="upper left", fontsize=9)

    # (b) gradient = learning signal
    ax_grad.plot(W, grad_sigmoid(W), color=SIGMOID_C, lw=2.2, label=r"Sigmoid: $\mathrm{d}x/\mathrm{d}w > 0$ everywhere")
    ax_grad.plot(W, grad_clip(W), color=CLIP_C, lw=2.2, ls="--", label=r"Clipping: $\mathrm{d}x/\mathrm{d}w = 0$ outside")
    ax_grad.axvspan(W[0], -(CENTRE - LO) / SLOPE, color=CLIP_C, alpha=0.08)
    ax_grad.axvspan((HI - CENTRE) / SLOPE, W[-1], color=CLIP_C, alpha=0.08)
    gpt_sig, = ax_grad.plot([], [], "o", color=SIGMOID_C, ms=9)
    gpt_clip, = ax_grad.plot([], [], "s", color=CLIP_C, ms=8)
    ax_grad.set_xlim(W[0], W[-1])
    ax_grad.set_ylim(-0.03, SLOPE * 1.8)
    ax_grad.set_xlabel(r"Network output $w$")
    ax_grad.set_ylabel(r"Gradient $\mathrm{d}x/\mathrm{d}w$ (m)")
    ax_grad.set_title("(b) Learning signal")
    ax_grad.legend(loc="upper left", fontsize=9)
    grad_text = ax_grad.text(0.0, SLOPE * 1.2, "", ha="center", fontsize=10, color=CLIP_C)

    # (c) decoded batch
    bins = np.linspace(LO - 0.05, HI + 0.05, 46)
    ax_hist.set_xlim(LO - 0.1, HI + 0.1)
    ax_hist.set_xlabel(r"Decoded position $x$ (m)")
    ax_hist.set_ylabel("Fraction of predictions")
    ax_hist.set_title("(c) Batch of predictions")
    hist_text = ax_hist.text(0.09, 0.78, "", transform=ax_hist.transAxes, va="top", fontsize=10)

    def draw(i):
        mu = path[i]
        pt_sig.set_data([mu], [decode_sigmoid(mu)])
        pt_clip.set_data([mu], [decode_clip(mu)])
        gpt_sig.set_data([mu], [grad_sigmoid(mu)])
        gpt_clip.set_data([mu], [grad_clip(mu)])
        grad_text.set_text("Clipped: no gradient,\nthe network cannot learn back" if grad_clip(mu) == 0 else "")

        w = mu + noise
        xs, xc = decode_sigmoid(w), decode_clip(w)
        for patch in list(ax_hist.patches):
            patch.remove()
        weights = np.full(w.size, 1.0 / w.size)
        ax_hist.hist(xs, bins=bins, weights=weights, color=SIGMOID_C, alpha=0.55, label="Sigmoid decoding")
        ax_hist.hist(xc, bins=bins, weights=weights, color=CLIP_C, alpha=0.55, histtype="stepfilled",
                     label="Hard clipping")
        on_edge = np.mean((xc <= LO) | (xc >= HI))
        hist_text.set_text(f"Clipping: {100 * on_edge:.0f}% exactly on the plate edge\n"
                           f"Sigmoid: {100 * np.mean((xs <= LO) | (xs >= HI)):.0f}% on the edge, all inside")
        ax_hist.set_ylim(0, 0.6)
        if i == 0:
            ax_hist.legend(loc="upper left", fontsize=9)
        for b in (LO, HI):
            ax_hist.axvline(b, color="0.4", lw=0.8, ls=":")
        return pt_sig, pt_clip, gpt_sig, gpt_clip

    fig.tight_layout()
    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps)
    out = Path(out)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    draw(int(np.argmin(np.abs(path[: n_frames // 2] - 2.5))))  # mean just past the clip limit: both effects visible
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved {out} and {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
