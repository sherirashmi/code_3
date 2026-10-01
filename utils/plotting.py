"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Plotting
Description : Visualization and plot-output utilities
==================================================

Every figure is rendered with the thesis LaTeX style from
``utils.plot_style`` (imported below, which applies it) and written through
``utils.plot_style.save_figure``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation, MaxNLocator, NullFormatter

from utils.plot_style import save_figure  # importing applies the LaTeX thesis style

import utils.physics as _physics
from utils.physics import Lx, Ly, X_grid, Y_grid, phi_mn, xf, yf

DEFAULT_PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

# Shared axis labels (LaTeX / mathtext formatted).
FREQ_LABEL = r"Frequency $f$ (Hz)"
ERP_LABEL = r"ERP (dB re $1\,\mathrm{pW}$)"

# One fixed colour per model name so the same model has the same colour in
# every comparison figure of the thesis.
_PALETTE = plt.get_cmap("tab10").colors + plt.get_cmap("Dark2").colors


def _tidy_loss_axes(ax) -> None:
    """Integer epoch ticks; on log axes label only full decades (avoids
    '3.6 x 10^0'-style minor labels when the loss spans < 1 decade)."""
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        lo, hi = ax.get_ylim()
        if hi / max(lo, 1e-300) < 10:  # less than one decade: plain numbers read better
            ax.set_yscale("linear")


def model_colors(names) -> dict[str, tuple]:
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(names)}


# ==================================================
# Common output helpers
# ==================================================


def _safe_name(name: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name)
    )
    return cleaned.strip("_") or "plot"


def operator_plot_dir(
    operator_name: str,
    plots_dir: str | Path | None = None,
) -> Path:
    """Create and return ``<plots_dir>/<operator>``."""
    base = Path(plots_dir) if plots_dir is not None else DEFAULT_PLOTS_DIR
    path = base / _safe_name(operator_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finalize_figure(
    fig,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Save/display a Matplotlib figure and always close it afterwards.

    ``show=False`` is non-blocking and is used by batch/all-model
    experiments. The figure is still saved whenever ``save_path`` is given.
    """
    if save_path is not None:
        save_figure(fig, save_path, close=False)
    if show:
        plt.show()
    plt.close(fig)


# ==================================================
# Loss Curves
# ==================================================


def plot_loss_curves(
    train_losses: list[float] | np.ndarray,
    val_losses: list[float] | np.ndarray,
    title: str = "Loss curves",
    ylabel: str = "Loss",
    *,
    log_y: bool = False,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    train_losses = np.asarray(train_losses, dtype=float)
    val_losses = np.asarray(val_losses, dtype=float)
    # A log axis cannot show non-positive values (e.g. NLL losses).
    if log_y and (np.nanmin(np.concatenate([train_losses, val_losses])) <= 0):
        log_y = False

    fig, ax = plt.subplots(figsize=(7, 4.3))
    plot_fn = ax.semilogy if log_y else ax.plot
    plot_fn(np.arange(1, train_losses.size + 1), train_losses, lw=1.8, label="Training")
    plot_fn(np.arange(1, val_losses.size + 1), val_losses, lw=1.8, ls="--", label="Validation")
    if val_losses.size and np.isfinite(val_losses).any():
        best = int(np.nanargmin(val_losses))
        ax.plot(best + 1, val_losses[best], "o", ms=5, color="black", zorder=5,
                label=f"Best validation (epoch {best + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both")
    _tidy_loss_axes(ax)
    ax.legend()
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_staged_loss_curves(
    stage_histories: Mapping[str, Mapping[str, list[float]]],
    title: str,
    *,
    ylabel: str = "Loss",
    log_y: bool = True,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Concatenated train/val curves of a multi-stage schedule on one epoch
    axis, with shaded/labelled stage boundaries (e.g. iFNO's 3 stages)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    offset = 0
    any_nonpositive = False
    shades = ["0.93", "white"]
    for i, (stage, hist) in enumerate(stage_histories.items()):
        tr = np.asarray(hist.get("train", []), dtype=float)
        va = np.asarray(hist.get("val", []), dtype=float)
        n = max(tr.size, va.size)
        if n == 0:
            continue
        any_nonpositive |= bool(np.nanmin(np.concatenate([tr, va])) <= 0)
        x = np.arange(offset + 1, offset + n + 1)
        ax.axvspan(offset + 0.5, offset + n + 0.5, color=shades[i % 2], zorder=0)
        ax.text(offset + 0.5 + n / 2, 1.01, stage, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9)
        marker = "o" if n < 10 else None  # keep very short stages visible
        ax.plot(x[: tr.size], tr, lw=1.8, color="C0", marker=marker, ms=4, label="Training" if i == 0 else None)
        ax.plot(x[: va.size], va, lw=1.8, ls="--", color="C1", marker=marker, ms=4,
                label="Validation" if i == 0 else None)
        offset += n
    if log_y and not any_nonpositive:
        ax.set_yscale("log")
    _tidy_loss_axes(ax)
    ax.set_xlabel("Epoch (cumulative over stages)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_all_models_loss(
    histories: Mapping[str, Mapping[str, list[float]]],
    *,
    title: str,
    ylabel: str = "Loss",
    log_y: bool = True,
    save_path: str | Path,
    show: bool = False,
) -> None:
    """Two panels (training | validation), one line per model."""
    names = list(histories)
    colors = model_colors(names)
    values = [np.asarray(histories[n].get(k, []), dtype=float) for n in names for k in ("train", "val")]
    values = [v for v in values if v.size]
    if log_y and values and min(np.nanmin(v) for v in values) <= 0:
        log_y = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, key, label in zip(axes, ("train", "val"), ("Training", "Validation")):
        for name in names:
            v = np.asarray(histories[name].get(key, []), dtype=float)
            if v.size:
                ax.plot(np.arange(1, v.size + 1), v, lw=1.6, color=colors[name], label=name)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(f"{label} loss")
        ax.grid(True, which="both")
        _tidy_loss_axes(ax)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(ncol=2 if len(names) > 5 else 1)
    fig.suptitle(title)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


# ==================================================
# ERP Spectrum
# ==================================================


def _mark_resonator_tuning_lines(ax, configuration: np.ndarray) -> None:
    """Draw one red dotted vertical line per resonator at its tuning frequency."""
    configuration = np.asarray(configuration, dtype=np.float64).reshape(-1, 5)
    for i, (_m, _k, f_t, _x, _y) in enumerate(configuration):
        ax.axvline(
            f_t,
            color="red",
            ls=":",
            lw=1.3,
            alpha=0.85,
            zorder=0,
            label=r"Resonator $f_t$" if i == 0 else None,
        )


def _draw_plate_layout(ax, configuration: np.ndarray) -> None:
    """Draw the plate outline with resonator and force positions."""
    configuration = np.asarray(configuration, dtype=np.float64).reshape(-1, 5)

    ax.add_patch(
        plt.Rectangle((0, 0), Lx, Ly, fill=False, edgecolor="black", lw=1.5)
    )
    ax.scatter([xf], [yf], marker="*", s=180, color="cyan", edgecolors="black",
               linewidths=0.8, zorder=3, label=r"Force $F_0$")
    ax.scatter(configuration[:, 3], configuration[:, 4], marker="o", s=90,
               color="crimson", edgecolors="black", linewidths=0.8, zorder=3,
               label="Resonator")
    for _m, _k, f_t, x, y in configuration:
        ax.annotate(
            f"{f_t:.1f} Hz",
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    pad_x, pad_y = 0.05 * Lx, 0.05 * Ly
    ax.set_xlim(-pad_x, Lx + pad_x)
    ax.set_ylim(-pad_y, Ly + pad_y)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title("Resonator layout on plate")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)


def plot_erp(
    freq_values: np.ndarray,
    erp: np.ndarray,
    title: str = "ERP spectrum",
    *,
    configuration: np.ndarray | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    if configuration is not None:
        fig, (ax, ax_plate) = plt.subplots(
            1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(freq_values, erp, lw=1.8, label="ERP")
    ax.set_xlabel(FREQ_LABEL)
    ax.set_ylabel(ERP_LABEL)
    ax.set_title(title)
    ax.grid(True)

    if configuration is not None:
        _mark_resonator_tuning_lines(ax, configuration)
        _draw_plate_layout(ax_plate, configuration)
    ax.legend()
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_erp_comparison(
    freq_values: np.ndarray,
    true_curve: np.ndarray,
    pred_curve: np.ndarray,
    *,
    title: str | None = None,
    configuration: np.ndarray | None = None,
    highlight_band: tuple[float, float] | None = None,
    highlight_label: str = "Held-out band (never trained on)",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    true_curve = np.asarray(true_curve)
    pred_curve = np.asarray(pred_curve)
    mse = float(np.mean((true_curve - pred_curve) ** 2))
    mae = float(np.mean(np.abs(true_curve - pred_curve)))

    if configuration is not None:
        fig, (ax, ax_plate) = plt.subplots(
            1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 4.8))

    if highlight_band is not None:
        ax.axvspan(highlight_band[0], highlight_band[1], color="orange", alpha=0.15, label=highlight_label)
    ax.plot(freq_values, true_curve, lw=2.2, color="black", label="Ground truth (solver)")
    ax.plot(freq_values, pred_curve, "--", lw=1.7, color="#C44E52", label="Prediction")
    ax.set_xlabel(FREQ_LABEL)
    ax.set_ylabel(ERP_LABEL)
    metrics_text = f"MSE $= {mse:.3f}$ dB$^2$, MAE $= {mae:.3f}$ dB"
    ax.set_title(f"{title}\n{metrics_text}" if title else metrics_text)
    ax.grid(True)

    if configuration is not None:
        _mark_resonator_tuning_lines(ax, configuration)
        _draw_plate_layout(ax_plate, configuration)
    ax.legend()
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


# ==================================================
# Point Displacement Spectrum
# ==================================================


def plot_displacement_spectrum(
    freq_values: np.ndarray,
    true_w: np.ndarray,
    pred_w: np.ndarray | None = None,
    title: str = "Complex displacement at measurement point",
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot real, imaginary, and magnitude point displacement versus frequency."""
    true_w = np.asarray(true_w)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(freq_values, np.real(true_w), lw=1.8, label="Ground truth")
    axes[1].plot(freq_values, np.imag(true_w), lw=1.8, label="Ground truth")
    axes[2].plot(freq_values, np.abs(true_w), lw=1.8, label="Ground truth")

    if pred_w is not None:
        pred_w = np.asarray(pred_w)
        axes[0].plot(freq_values, np.real(pred_w), "--", lw=1.6, label="Prediction")
        axes[1].plot(freq_values, np.imag(pred_w), "--", lw=1.6, label="Prediction")
        axes[2].plot(freq_values, np.abs(pred_w), "--", lw=1.6, label="Prediction")

    axes[0].set_ylabel(r"$\mathrm{Re}(w)$")
    axes[1].set_ylabel(r"$\mathrm{Im}(w)$")
    axes[2].set_ylabel(r"$|w|$")
    axes[2].set_xlabel(FREQ_LABEL)

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


# ==================================================
# Resonator Field
# ==================================================


def plot_resonator_field(
    field: np.ndarray,
    resonators: list[dict[str, float]] | None = None,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    contour = ax.contourf(X_grid, Y_grid, field, levels=50)
    fig.colorbar(contour, ax=ax)
    if resonators is not None:
        for resonator in resonators:
            ax.scatter(resonator["x"], resonator["y"], s=80)
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title("Resonator field")
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


# ==================================================
# Displacement Field
# ==================================================


def _select_complex_component(field: np.ndarray, component: str) -> np.ndarray:
    component = component.lower()
    if component == "real":
        return np.real(field)
    if component == "imag":
        return np.imag(field)
    if component in {"abs", "magnitude"}:
        return np.abs(field)
    raise ValueError("component must be 'real', 'imag', 'abs', or 'magnitude'.")


def plot_displacement(
    displacement: np.ndarray,
    title: str = "Displacement field",
    component: str = "magnitude",
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    display_field = _select_complex_component(displacement, component)
    if display_field.ndim != 2:
        raise ValueError("plot_displacement expects a 2D field for one frequency.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(
        display_field,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, label=f"Displacement ({component})")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    ax.set_title(title)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_displacement_comparison(
    true_field: np.ndarray,
    pred_field: np.ndarray,
    title: str = "",
    component: str = "magnitude",
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    true_display = _select_complex_component(true_field, component)
    pred_display = _select_complex_component(pred_field, component)
    if true_display.shape != pred_display.shape or true_display.ndim != 2:
        raise ValueError("true_field and pred_field must be matching 2D fields.")

    error = pred_display - true_display
    vmax = max(np.max(np.abs(true_display)), np.max(np.abs(pred_display)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(true_display, vmin=-vmax, vmax=vmax)
    axes[0].set_title("Ground truth")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pred_display, vmin=-vmax, vmax=vmax)
    axes[1].set_title("Prediction")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(error)
    axes[2].set_title("Error")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle(title)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


# ==================================================
# Scatter Comparison
# ==================================================


def plot_prediction_scatter(
    targets: np.ndarray,
    predictions: np.ndarray,
    xlabel: str = "Ground truth",
    ylabel: str = "Prediction",
    title: str = "Prediction vs ground truth",
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    max_points: int = 200_000,
) -> None:
    targets = np.asarray(targets).ravel()
    predictions = np.asarray(predictions).ravel()
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape.")

    mn = min(float(targets.min()), float(predictions.min()))
    mx = max(float(targets.max()), float(predictions.max()))

    t64, p64 = targets.astype(np.float64), predictions.astype(np.float64)
    ss_tot = float(np.sum((t64 - t64.mean()) ** 2))
    r2 = 1.0 - float(np.sum((p64 - t64) ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    # Display a random subsample for very large test sets (100k spectra x 301
    # frequencies = 3e6 points) -- statistics above still use every point.
    if targets.size > max_points:
        pick = np.random.default_rng(0).choice(targets.size, size=max_points, replace=False)
        targets, predictions = targets[pick], predictions[pick]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(targets, predictions, alpha=0.25, s=4, edgecolors="none", rasterized=True)
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label=r"Ideal: $y = x$")
    ax.text(0.04, 0.96, f"$R^2 = {r2:.4f}$", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


# ==================================================
# Mode Shapes
# ==================================================


def plot_mode_shape(
    mode_index: int,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot one plate mode of the CURRENT modal basis (dataset dependent,
    see ``utils.physics.set_modal_resolution``)."""
    modes = _physics.modes
    if not 0 <= mode_index < len(modes):
        raise IndexError(f"mode_index must be in [0, {len(modes) - 1}].")

    m = int(modes[mode_index, 1])
    n = int(modes[mode_index, 2])
    Z = phi_mn(m, n, X_grid, Y_grid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    contour = ax.contourf(X_grid, Y_grid, Z, levels=50)
    fig.colorbar(contour, ax=ax)
    ax.set_title(f"Mode $({m}, {n})$, $f = {modes[mode_index, 0] / (2 * np.pi):.1f}$ Hz")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


# ==================================================
# Experiment plot bundle
# ==================================================


def save_operator_experiment_plots(
    operator_name: str,
    *,
    history: dict[str, list[float]] | None = None,
    metrics: dict[str, object] | None = None,
    frequency_values: np.ndarray | None = None,
    prediction: dict[str, object] | None = None,
    plot_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    num_configurations: int = 5,
    title_suffix: str = "",
    show: bool = False,
) -> Path:
    """Save every plot belonging to one operator experiment into ONE folder.

    ``plot_dir`` is that model's own folder (preferred; see
    ``utils.paths.forward_plot_dir``). ``plots_dir`` is the legacy form:
    the parent folder, with ``<operator_name>/`` appended.

    Files written:
      - ``loss_curve.png``                       (if ``history``)
      - ``erp_spectrum_test_config_01..NN.png``  (if ``metrics``)
      - ``prediction_vs_ground_truth.png``       (if ``metrics``)
      - ``prediction_spectrum.png``              (if ``prediction``)
    """
    plot_dir = Path(plot_dir) if plot_dir is not None else operator_plot_dir(operator_name, plots_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    label = f"{operator_name}{title_suffix}"

    expected: list[Path] = []

    if history is not None:
        train_losses = history.get("train", [])
        val_losses = history.get("val", [])
        if len(train_losses) and len(val_losses):
            plot_loss_curves(
                train_losses,
                val_losses,
                title=f"{label}: training history",
                ylabel="Normalised ERP loss",
                log_y=True,
                save_path=plot_dir / "loss_curve.png",
                show=show,
            )
            expected.append(plot_dir / "loss_curve.png")

    if metrics is not None:
        if frequency_values is None:
            raise ValueError(
                "frequency_values are required when saving evaluation plots."
            )
        pred = np.asarray(metrics["predictions"])
        true = np.asarray(metrics["targets"])
        freq = np.asarray(frequency_values, dtype=np.float64)
        if pred.shape != true.shape:
            raise ValueError("Evaluation predictions and targets must have matching shapes.")
        if pred.ndim != 2 or pred.shape[1] != freq.size:
            raise ValueError(
                "Evaluation arrays must have shape (n_configurations, n_frequencies)."
            )

        configurations = metrics.get("configurations")
        if configurations is not None:
            configurations = np.asarray(configurations)

        n_to_plot = min(max(int(num_configurations), 0), true.shape[0])
        for i in range(n_to_plot):
            path = plot_dir / f"erp_spectrum_test_config_{i + 1:02d}.png"
            plot_erp_comparison(
                freq,
                true[i],
                pred[i],
                title=f"{label}: test configuration {i + 1}",
                configuration=configurations[i] if configurations is not None else None,
                save_path=path,
                show=show,
            )
            expected.append(path)

        plot_prediction_scatter(
            true,
            pred,
            xlabel=f"True {ERP_LABEL}",
            ylabel=f"Predicted {ERP_LABEL}",
            title=f"{label}: predicted vs true ERP (test set)",
            save_path=plot_dir / "prediction_vs_ground_truth.png",
            show=show,
        )
        expected.append(plot_dir / "prediction_vs_ground_truth.png")

    if prediction is not None:
        freq = np.asarray(prediction["frequencies"], dtype=np.float64)
        pred = np.asarray(prediction["prediction"])
        ground_truth = prediction.get("ground_truth")
        prediction_configuration = prediction.get("configuration")
        if ground_truth is None:
            plot_erp(
                freq,
                pred,
                title=f"{label}: predicted ERP spectrum",
                configuration=prediction_configuration,
                save_path=plot_dir / "prediction_spectrum.png",
                show=show,
            )
        else:
            plot_erp_comparison(
                freq,
                np.asarray(ground_truth),
                pred,
                title=f"{label}: solver vs neural-operator ERP",
                configuration=prediction_configuration,
                save_path=plot_dir / "prediction_spectrum.png",
                show=show,
            )
        expected.append(plot_dir / "prediction_spectrum.png")

    # Verify every plot requested by this call exists before returning.
    missing = [path for path in expected if not path.exists() or path.stat().st_size <= 0]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise OSError(f"Plot saving verification failed. Missing/empty files:\n{formatted}")

    print(f"Verified {len(expected)} plot file(s) in: {plot_dir}")
    return plot_dir


# ==================================================
# Cross-architecture comparison plots
# ==================================================


# Fixed categorical order (validated colour-blind-safe adjacent slots) for
# the error-breakdown categories; colour follows the category, never rank.
_BREAKDOWN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def plot_error_breakdown(
    rows: list[dict[str, object]],
    *,
    save_path: str | Path | None = None,
    title_suffix: str = "",
    show: bool = False,
) -> None:
    """Where each model's test error sits: frequency bands, peaks, hard cases.

    ``rows`` need the ``spectrum_error_diagnostics`` keys plus ``model`` and
    ``rmse``. Three panels with their own y-axes (the at-peak error is
    several times the band errors), each a grouped bar chart per model.
    """
    names = [str(r["model"]) for r in rows]
    panels = (
        ("Frequency bands", (
            ("rmse_low_band_db", "Lowest 5% of $f$"),
            ("rmse_interior_db", "Interior"),
            ("rmse_high_band_db", "Highest 5% of $f$"),
        )),
        ("Resonance peaks", (
            ("rmse_at_peaks_db", "At true peaks"),
            ("rmse_off_peaks_db", "Off peaks"),
        )),
        ("Hard-case strata (10% each)", (
            ("rmse_close_ft_db", r"Closest $f_t$ pair"),
            ("rmse_near_edge_db", "Nearest plate edge"),
            ("rmse", "All test spectra"),
        )),
    )
    x = np.arange(len(names), dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(max(8.0, 0.9 * len(names) + 3.0), 10.5), sharex=True)
    for ax, (title, series) in zip(axes, panels):
        width = 0.8 / len(series)
        for j, (key, label) in enumerate(series):
            values = np.array([float(r[key]) for r in rows])
            ax.bar(
                x + (j - (len(series) - 1) / 2) * width, values, width,
                color=_BREAKDOWN_COLORS[j], edgecolor="white", linewidth=1.0, label=label,
            )
        ax.set_ylabel("RMSE (dB)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", ncol=len(series), fontsize=9)
        ax.margins(y=0.18)
    axes[-1].set_xticks(x, names)
    axes[-1].set_xlabel("Architecture")
    fig.suptitle(f"Test-error breakdown{title_suffix}")
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def save_all_model_comparison_plots(
    plot_data: dict[str, dict[str, np.ndarray]],
    comparison_rows: list[dict[str, object]],
    *,
    out_dir: str | Path | None = None,
    plots_dir: str | Path | None = None,
    title_suffix: str = "",
    show: bool = False,
) -> Path:
    """Save plots that compare every trained operator in one figure.

    ``out_dir`` is the ALL_MODELS folder itself (preferred); ``plots_dir``
    is the legacy form with ``ALL_MODELS`` appended.
    """
    if out_dir is None:
        base = Path(plots_dir) if plots_dir is not None else DEFAULT_PLOTS_DIR
        out_dir = base / "ALL_MODELS"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(plot_data)
    if not model_names:
        raise ValueError("plot_data is empty.")
    colors = model_colors(model_names)
    suffix = title_suffix

    # 1+2) Training and validation loss of every model, one line per model.
    for key, label, filename in (
        ("train_loss", "Training", "all_training_loss.png"),
        ("val_loss", "Validation", "all_validation_loss.png"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for name in model_names:
            values = np.asarray(plot_data[name][key], dtype=float)
            ax.semilogy(np.arange(1, len(values) + 1), values, lw=1.6, color=colors[name], label=name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Normalised {label.lower()} loss")
        ax.set_title(f"{label} loss: all neural operators{suffix}")
        ax.grid(True, which="both")
        _tidy_loss_axes(ax)
        ax.legend(ncol=2)
        fig.tight_layout()
        _finalize_figure(fig, save_path=out_dir / filename, show=show)

    # 3) Both side by side in one figure (thesis-ready single figure).
    plot_all_models_loss(
        {n: {"train": plot_data[n]["train_loss"], "val": plot_data[n]["val_loss"]} for n in model_names},
        title=f"Training history: all neural operators{suffix}",
        ylabel="Normalised ERP loss",
        save_path=out_dir / "all_models_loss.png",
        show=show,
    )

    def _boxplot(metric_key: str, ylabel: str, title: str, filename: str) -> None:
        values = [
            np.asarray(plot_data[name][metric_key], dtype=float)
            for name in model_names
        ]
        values = [v[np.isfinite(v)] for v in values]
        fig, ax = plt.subplots(figsize=(9, 5))
        box = ax.boxplot(values, showmeans=True, showfliers=False, patch_artist=True)
        for patch, name in zip(box["boxes"], model_names):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.55)
        ax.set_xticks(np.arange(1, len(model_names) + 1), model_names)
        ax.set_xlabel("Architecture")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}{suffix}")
        ax.grid(True, axis="y")
        fig.tight_layout()
        _finalize_figure(fig, save_path=out_dir / filename, show=show)

    # Distribution plots across complete test spectra.
    _boxplot("spectrum_rmse", "RMSE (dB)", "Per-spectrum test RMSE", "test_rmse_boxplot.png")
    _boxplot("spectrum_mae", "MAE (dB)", "Per-spectrum test MAE", "test_mae_boxplot.png")
    _boxplot("spectrum_pearson", r"Pearson $r$", "Per-spectrum Pearson correlation",
             "spectrum_pearson_boxplot.png")
    _boxplot("peak_frequency_abs_error", r"$|\Delta f_{\mathrm{peak}}|$ (Hz)",
             "Dominant-peak frequency error", "peak_frequency_error_boxplot.png")
    _boxplot("peak_amplitude_abs_error", "Absolute ERP error at true peak (dB)",
             "ERP error at true peak", "peak_amplitude_error_boxplot.png")

    # Overall test RMSE / MAE as grouped bars (same physical unit: dB).
    row_by_model = {str(row["model"]): row for row in comparison_rows}
    x = np.arange(len(model_names), dtype=float)
    width = 0.36
    rmse = np.array([float(row_by_model[name]["rmse"]) for name in model_names])
    mae = np.array([float(row_by_model[name]["mae"]) for name in model_names])

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, rmse, width, label="RMSE")
    b2 = ax.bar(x + width / 2, mae, width, label="MAE")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x, model_names)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Error (dB)")
    ax.set_title(f"Overall test error{suffix}")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    _finalize_figure(fig, save_path=out_dir / "overall_test_error_bars.png", show=show)

    # Dimensionless agreement metrics in one grouped-bar graph.
    width = 0.25
    global_r = np.array([float(row_by_model[name]["pearson_global"]) for name in model_names])
    mean_r = np.array([float(row_by_model[name]["pearson_mean"]) for name in model_names])
    r2 = np.array([float(row_by_model[name]["r2"]) for name in model_names])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, global_r, width, label=r"Global Pearson $r$")
    ax.bar(x, mean_r, width, label=r"Mean per-spectrum $r$")
    ax.bar(x + width, r2, width, label=r"$R^2$")
    ax.set_xticks(x, model_names)
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Score")
    lo = min(0.0, float(np.nanmin(np.concatenate([global_r, mean_r, r2]))))
    ax.set_ylim(lo, 1.05)
    ax.set_title(f"Prediction agreement{suffix}")
    ax.grid(True, axis="y")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _finalize_figure(fig, save_path=out_dir / "overall_agreement_bars.png", show=show)

    if all("rmse_at_peaks_db" in row for row in comparison_rows):
        plot_error_breakdown(
            [row_by_model[name] for name in model_names],
            save_path=out_dir / "error_breakdown_bars.png",
            title_suffix=suffix,
            show=show,
        )

    print(f"Saved all-model comparison plots to: {out_dir}")
    return out_dir


__all__ = [
    "DEFAULT_PLOTS_DIR",
    "FREQ_LABEL",
    "ERP_LABEL",
    "model_colors",
    "operator_plot_dir",
    "plot_loss_curves",
    "plot_staged_loss_curves",
    "plot_all_models_loss",
    "plot_erp",
    "plot_erp_comparison",
    "plot_displacement_spectrum",
    "plot_resonator_field",
    "plot_displacement",
    "plot_displacement_comparison",
    "plot_prediction_scatter",
    "plot_mode_shape",
    "save_operator_experiment_plots",
    "plot_error_breakdown",
    "save_all_model_comparison_plots",
    "save_figure",
]
