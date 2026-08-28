"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Plotting
Description : Visualization and plot-output utilities
==================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.physics import Lx, Ly, X_grid, Y_grid, modes, phi_mn

DEFAULT_PLOTS_DIR = Path.cwd() / "plots"


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
    """Create and return ``plots/<operator>`` beside the project code."""
    base = Path(plots_dir) if plots_dir is not None else DEFAULT_PLOTS_DIR
    path = base / _safe_name(operator_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finalize_figure(
    fig,
    *,
    save_path: str | Path | None = None,
    show: bool = True,
    dpi: int = 200,
) -> None:
    """Save/display a Matplotlib figure and always close it afterwards.

    ``show=False`` is intentionally non-blocking and is used by batch/all-model
    experiments.  The figure is still saved whenever ``save_path`` is supplied.
    """
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if not path.exists() or path.stat().st_size <= 0:
            raise OSError(f"Matplotlib did not create a valid plot file: {path}")
        print(f"Plot saved: {path.resolve()} ({path.stat().st_size / 1024:.1f} KiB)")
    if show:
        plt.show()
    plt.close(fig)


# ==================================================
# Loss Curves
# ==================================================


def plot_loss_curves(
    train_losses: list[float] | np.ndarray,
    val_losses: list[float] | np.ndarray,
    title: str = "Loss Curves",
    ylabel: str = "Loss",
    *,
    log_y: bool = False,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_fn = ax.semilogy if log_y else ax.plot
    plot_fn(train_losses, lw=2, label="Training")
    plot_fn(val_losses, lw=2, label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


# ==================================================
# ERP Spectrum
# ==================================================


def plot_erp(
    freq_values: np.ndarray,
    erp: np.ndarray,
    title: str = "ERP Spectrum",
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freq_values, erp, lw=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ERP (dB)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)


def plot_erp_comparison(
    freq_values: np.ndarray,
    true_curve: np.ndarray,
    pred_curve: np.ndarray,
    *,
    title: str | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    true_curve = np.asarray(true_curve)
    pred_curve = np.asarray(pred_curve)
    mse = np.mean((true_curve - pred_curve) ** 2)
    mae = np.mean(np.abs(true_curve - pred_curve))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_values, true_curve, lw=3, label="Ground Truth")
    ax.plot(freq_values, pred_curve, "--", lw=2, label="Prediction")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ERP (dB)")
    ax.set_title(title or f"MSE = {mse:.3f} | MAE = {mae:.3f}")
    ax.grid(True)
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

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(freq_values, np.real(true_w), lw=2, label="Ground Truth")
    axes[1].plot(freq_values, np.imag(true_w), lw=2, label="Ground Truth")
    axes[2].plot(freq_values, np.abs(true_w), lw=2, label="Ground Truth")

    if pred_w is not None:
        pred_w = np.asarray(pred_w)
        axes[0].plot(freq_values, np.real(pred_w), "--", lw=2, label="Prediction")
        axes[1].plot(freq_values, np.imag(pred_w), "--", lw=2, label="Prediction")
        axes[2].plot(freq_values, np.abs(pred_w), "--", lw=2, label="Prediction")

    axes[0].set_ylabel("Real(w)")
    axes[1].set_ylabel("Imag(w)")
    axes[2].set_ylabel("|w|")
    axes[2].set_xlabel("Frequency (Hz)")

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
    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.contourf(X_grid, Y_grid, field, levels=50)
    fig.colorbar(contour, ax=ax)
    if resonators is not None:
        for resonator in resonators:
            ax.scatter(resonator["x"], resonator["y"], s=80)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Resonator Field")
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
    title: str = "Displacement Field",
    component: str = "magnitude",
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    display_field = _select_complex_component(displacement, component)
    if display_field.ndim != 2:
        raise ValueError("plot_displacement expects a 2D field for one frequency.")

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        display_field,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, label=f"Displacement ({component})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
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
    axes[0].set_title("Ground Truth")
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
    xlabel: str = "Ground Truth",
    ylabel: str = "Prediction",
    title: str = "Prediction vs Ground Truth",
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    targets = np.asarray(targets).ravel()
    predictions = np.asarray(predictions).ravel()
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape.")

    mn = min(float(targets.min()), float(predictions.min()))
    mx = max(float(targets.max()), float(predictions.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, predictions, alpha=0.5, s=10)
    ax.plot([mn, mx], [mn, mx], "--", label="Ideal: y = x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
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
    if not 0 <= mode_index < len(modes):
        raise IndexError(f"mode_index must be in [0, {len(modes) - 1}].")

    m = int(modes[mode_index, 1])
    n = int(modes[mode_index, 2])
    Z = phi_mn(m, n, X_grid, Y_grid)

    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.contourf(X_grid, Y_grid, Z, levels=50)
    fig.colorbar(contour, ax=ax)
    ax.set_title(f"Mode ({m}, {n})")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
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
    plots_dir: str | Path | None = None,
    num_configurations: int = 5,
    show: bool = False,
) -> Path:
    """Save all plots belonging to one operator experiment.

    This is the public plotting entry point used by ``main.py``.  The main
    workflow calls this function explicitly after training/evaluation, so plot
    persistence does not depend on whether lower-level training code happens to
    invoke plotting.
    """
    plot_dir = operator_plot_dir(operator_name, plots_dir)

    if history is not None:
        train_losses = history.get("train", [])
        val_losses = history.get("val", [])
        if len(train_losses) and len(val_losses):
            plot_loss_curves(
                train_losses,
                val_losses,
                title=f"{operator_name} ERP training loss",
                ylabel="Normalized loss",
                log_y=True,
                save_path=plot_dir / "loss_curve.png",
                show=show,
            )

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

        n_to_plot = min(max(int(num_configurations), 0), true.shape[0])
        for i in range(n_to_plot):
            plot_erp_comparison(
                freq,
                true[i],
                pred[i],
                title=f"{operator_name} - test configuration {i + 1}",
                save_path=plot_dir / f"erp_spectrum_test_config_{i + 1:02d}.png",
                show=show,
            )

        plot_prediction_scatter(
            true,
            pred,
            xlabel="Ground Truth ERP (dB)",
            ylabel="Predicted ERP (dB)",
            title=f"{operator_name} - test ERP prediction vs ground truth",
            save_path=plot_dir / "prediction_vs_ground_truth.png",
            show=show,
        )

    if prediction is not None:
        freq = np.asarray(prediction["frequencies"], dtype=np.float64)
        pred = np.asarray(prediction["prediction"])
        ground_truth = prediction.get("ground_truth")
        if ground_truth is None:
            plot_erp(
                freq,
                pred,
                title=f"{operator_name} - predicted ERP spectrum",
                save_path=plot_dir / "prediction_spectrum.png",
                show=show,
            )
        else:
            plot_erp_comparison(
                freq,
                np.asarray(ground_truth),
                pred,
                title=f"{operator_name} - solver vs neural-operator ERP spectrum",
                save_path=plot_dir / "prediction_spectrum.png",
                show=show,
            )

    # Verify every plot requested by this call exists before returning.
    expected: list[Path] = []
    if history is not None and len(history.get("train", [])) and len(history.get("val", [])):
        expected.append(plot_dir / "loss_curve.png")
    if metrics is not None:
        true = np.asarray(metrics["targets"])
        n_to_plot = min(max(int(num_configurations), 0), true.shape[0])
        expected.extend(
            plot_dir / f"erp_spectrum_test_config_{i + 1:02d}.png"
            for i in range(n_to_plot)
        )
        expected.append(plot_dir / "prediction_vs_ground_truth.png")
    if prediction is not None:
        expected.append(plot_dir / "prediction_spectrum.png")

    missing = [path for path in expected if not path.exists() or path.stat().st_size <= 0]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise OSError(f"Plot saving verification failed. Missing/empty files:\n{formatted}")

    print(f"Verified {len(expected)} plot file(s) in: {plot_dir.resolve()}")
    return plot_dir


__all__ = [
    "DEFAULT_PLOTS_DIR",
    "operator_plot_dir",
    "plot_loss_curves",
    "plot_erp",
    "plot_erp_comparison",
    "plot_displacement_spectrum",
    "plot_resonator_field",
    "plot_displacement",
    "plot_displacement_comparison",
    "plot_prediction_scatter",
    "plot_mode_shape",
    "save_operator_experiment_plots",
]
