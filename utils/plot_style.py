"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Plot style
Description : Thesis-wide LaTeX typography for every saved figure
==================================================

Importing this module applies one consistent, LaTeX-typeset look to every
Matplotlib figure the project produces (Computer Modern serif text and
maths, thesis-sized fonts, 300-dpi PDF-quality output).

Rendering mode, chosen once at import:

* **Real LaTeX** (``text.usetex = True``) when a ``latex`` executable is
  on PATH, so labels are typeset by the same engine as the thesis.
* **Matplotlib mathtext** with Matplotlib's bundled Computer Modern fonts
  otherwise -- visually the same font, no TeX installation required.

Override with the environment variable ``THESIS_USETEX=1`` (force LaTeX) or
``THESIS_USETEX=0`` (force mathtext).

Every figure should be written through :func:`save_figure`. In LaTeX mode it
first escapes TeX-special characters (``_ % & #`` and a few Unicode symbols)
that appear *outside* ``$...$`` in titles/labels/legends, and if TeX still
fails on some label it re-renders that figure with mathtext instead of
crashing a multi-hour training run at the plotting step.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.text import Text

SAVE_DPI = 300


def _latex_available() -> bool:
    forced = os.environ.get("THESIS_USETEX")
    if forced is not None:
        return forced.strip().lower() in {"1", "true", "yes", "on"}
    return shutil.which("latex") is not None and (
        shutil.which("dvipng") is not None or shutil.which("gs") is not None
    )


USETEX = _latex_available()

_BASE_RC = {
    # Typography: Computer Modern for text and maths.
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,  # cmr10 has no Unicode minus glyph
    # Thesis-friendly sizes.
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    # Lines / axes.
    "lines.linewidth": 1.6,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    # Output.
    "savefig.dpi": SAVE_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_thesis_style(usetex: bool | None = None) -> None:
    """(Re)apply the thesis rcParams. Safe to call repeatedly."""
    global USETEX
    if usetex is not None:
        USETEX = bool(usetex)
    matplotlib.rcParams.update(_BASE_RC)
    matplotlib.rcParams["text.usetex"] = USETEX
    if USETEX:
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"


# --------------------------------------------------------------------------
# LaTeX-safe text
# --------------------------------------------------------------------------

_UNICODE_TO_TEX = {
    "²": r"$^{2}$",
    "³": r"$^{3}$",
    "°": r"$^{\circ}$",
    "±": r"$\pm$",
    "×": r"$\times$",
    "→": r"$\rightarrow$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "≈": r"$\approx$",
    "−": "-",
    "–": "--",
    "—": "---",
}
_TEX_SPECIAL = re.compile(r"(?<!\\)([_%&#])")


def latex_safe(text: str) -> str:
    """Escape TeX-special characters outside ``$...$`` math segments."""
    if not text:
        return text
    parts = re.split(r"(?<!\\)(\$[^$]*\$)", text)
    out = []
    for part in parts:
        if part.startswith("$") and part.endswith("$") and len(part) >= 2:
            out.append(part)
            continue
        part = _TEX_SPECIAL.sub(r"\\\1", part)
        for uni, tex in _UNICODE_TO_TEX.items():
            part = part.replace(uni, tex)
        out.append(part)
    return "".join(out)


def _texts(fig) -> list[Text]:
    return [t for t in fig.findobj(Text) if t.get_text()]


def _escape_figure_text(fig) -> None:
    for t in _texts(fig):
        t.set_text(latex_safe(t.get_text()))


def _disable_usetex(fig) -> None:
    for t in fig.findobj(Text):
        t.set_usetex(False)
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_ticklabels():
                t.set_usetex(False)


def save_figure(fig, path: str | Path, *, dpi: int = SAVE_DPI, close: bool = True, verbose: bool = True) -> Path:
    """Save ``fig`` with the thesis style; never lets a TeX error kill a run.

    Creates parent folders, verifies a non-empty file was written, and closes
    the figure afterwards unless ``close=False``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        except Exception as exc:  # TeX failure on some label -> mathtext fallback
            if not USETEX:
                raise
            print(f"LaTeX rendering failed for {path.name} ({type(exc).__name__}); using mathtext fallback.")
            _disable_usetex(fig)
            with matplotlib.rc_context({"text.usetex": False}):
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if not path.exists() or path.stat().st_size <= 0:
            raise OSError(f"Matplotlib did not create a valid plot file: {path}")
        if verbose:
            print(f"Plot saved: {path} ({path.stat().st_size / 1024:.1f} KiB)")
    finally:
        if close:
            plt.close(fig)
    return path


def _probe_usetex() -> bool:
    """Render one tiny LaTeX string; False if the TeX toolchain is unusable."""
    import io

    fig = plt.figure(figsize=(1, 1))
    try:
        with matplotlib.rc_context({"text.usetex": True}):
            fig.text(0.5, 0.5, r"$R^2$ test")
            fig.savefig(io.BytesIO(), format="png")
        return True
    except Exception as exc:  # missing latex/dvipng, broken install, ...
        print(f"LaTeX not usable for Matplotlib ({type(exc).__name__}); using built-in Computer Modern mathtext.")
        return False
    finally:
        plt.close(fig)


_original_set_text = Text.set_text


def _latex_safe_set_text(self, s):
    """In LaTeX mode, escape TeX specials (outside $...$) in every text the
    moment it is set, so titles/labels/legends never break the TeX run."""
    if USETEX and isinstance(s, str):
        s = latex_safe(s)
    return _original_set_text(self, s)


if USETEX:
    USETEX = _probe_usetex()
if USETEX:
    Text.set_text = _latex_safe_set_text

apply_thesis_style()

__all__ = ["USETEX", "SAVE_DPI", "apply_thesis_style", "latex_safe", "save_figure"]
