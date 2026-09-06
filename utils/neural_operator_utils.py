"""
==================================================
Project     : Vibro-Acoustic Metamaterials
Module      : Neural Operator Utilities
Description : Shared ERP neural-operator training/evaluation workflow
==================================================

All operator models use the same convention:

    configuration : (batch, num_res, 3) normalized [f_t, x, y]
    frequency     : (batch, n_freq, 1) normalized evaluation frequencies
    output        : (batch, n_freq, 1) normalized ERP

Dataset preprocessing is owned by ``erp_dataset.py``. Plot construction and
file output are owned by ``plotting.py``. This module owns reusable network
blocks, full-spectrum loaders, training/evaluation and checkpoint workflows.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .erp_dataset import (
    DEFAULT_DATASET_FILE,
    configuration_to_resonators,
    denormalize_configuration_array,
    denormalize_erp_array,
    normalize_configuration_array,
    normalize_erp_array,
    normalize_frequency_array,
    prepare_erp_dataset,
)
from utils.physics import freqs, num_res as default_num_res
from .plotting import (
    operator_plot_dir,
    plot_erp_comparison,
    plot_loss_curves,
    plot_prediction_scatter,
)
from .solver import compute_erp_spectrum
from .support import device, seed_everything


# ==================================================
# Small reusable network blocks
# ==================================================


# Registry of activation choices every operator's own backbone can select
# from (the shared ResonatorSetEncoder/ResonanceQueryEncoder/
# FrequencyRefinement1d blocks keep their own fixed activations -- those are
# common "same information" tooling every architecture is equalized to have,
# not part of what makes one architecture different from another).
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "mish": nn.Mish,
}


def resolve_activation(activation: str | type[nn.Module]) -> type[nn.Module]:
    """Resolve an activation choice to its ``nn.Module`` subclass.

    Accepts either a name registered in ``ACTIVATIONS`` (the form stored in
    ``DEFAULT_MODEL_CONFIG`` so checkpoints stay plain-data serializable) or
    an ``nn.Module`` subclass passed straight through, so existing call sites
    that already pass a class (e.g. ``nn.SiLU``) keep working unchanged.
    """
    if isinstance(activation, str):
        key = activation.lower().strip()
        if key not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose one of: {sorted(ACTIVATIONS)}"
            )
        return ACTIVATIONS[key]
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation
    raise TypeError("activation must be a registered name or an nn.Module subclass.")


class MLP(nn.Module):
    """Compact fully connected network."""

    def __init__(
        self,
        widths: Sequence[int],
        activation: type[nn.Module] = nn.Tanh,
        final_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        widths = [int(v) for v in widths]
        if len(widths) < 2 or any(v <= 0 for v in widths):
            raise ValueError("widths must contain at least two positive integers.")

        layers: list[nn.Module] = []
        for i, (din, dout) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
            layers.append(nn.Linear(din, dout))
            if i < len(widths) - 2:
                layers.append(activation())
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLPBlock(nn.Module):
    """Two-layer residual MLP block."""

    def __init__(self, width: int, activation: type[nn.Module] = nn.Tanh) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            activation(),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x + self.net(x)))


class FrequencyRefinement1d(nn.Module):
    """Shared residual local frequency mixer used to sharpen ERP peaks.

    Every operator uses this exact module for its final cross-frequency
    mixing stage, so no architecture gets a stronger or weaker "peak
    sharpening" tool than any other purely as an implementation accident.
    Operates on ``(batch, width, n_freq)`` feature maps.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
        )
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(x + self.net(x)))


def physics_aware_resonator_features(
    configuration: torch.Tensor,
    harmonics: int = 4,
) -> torch.Tensor:
    """Augment normalized ``[f_t, x, y]`` with plate-inspired sine features.

    ``x`` and ``y`` are already normalized by ``Lx`` and ``Ly`` in the common
    preprocessing pipeline.  Therefore ``sin(m*pi*x)`` and ``sin(n*pi*y)``
    follow the same spatial structure that appears in the plate mode shapes.
    Low-order tensor-product terms are included so the network does not have to
    rediscover the dominant modal spatial interactions from raw coordinates.
    """
    if configuration.ndim < 2 or configuration.shape[-1] != 3:
        raise ValueError("configuration must end with normalized [f_t, x, y].")
    if harmonics < 0:
        raise ValueError("harmonics cannot be negative.")

    if harmonics == 0:
        return configuration

    f_t = configuration[..., 0:1]
    x = configuration[..., 1:2]
    y = configuration[..., 2:3]

    x_modes = [torch.sin(math.pi * float(m) * x) for m in range(1, harmonics + 1)]
    y_modes = [torch.sin(math.pi * float(n) * y) for n in range(1, harmonics + 1)]
    products = [xm * yn for xm in x_modes for yn in y_modes]
    return torch.cat([f_t, x, y, *x_modes, *y_modes, *products], dim=-1)


class ResonatorSetEncoder(nn.Module):
    """Permutation-invariant, physics-aware encoder for resonator sets.

    Every resonator is embedded with shared weights, using normalized raw
    ``[f_t,x,y]`` plus low-order plate-inspired spatial sine features.  Mean and
    max pooling preserve permutation invariance while retaining both distributed
    and dominant-resonator information.
    """

    def __init__(
        self,
        feature_dim: int = 3,
        hidden_dim: int = 128,
        element_dim: int = 128,
        output_dim: int = 128,
        modal_harmonics: int = 4,
    ) -> None:
        super().__init__()
        if int(feature_dim) != 3:
            raise ValueError("ResonatorSetEncoder expects [f_t,x,y] feature_dim=3.")
        self.modal_harmonics = int(modal_harmonics)
        augmented_dim = 3 + 2 * self.modal_harmonics + self.modal_harmonics**2
        self.element_net = MLP(
            [augmented_dim, hidden_dim, hidden_dim, element_dim],
            activation=nn.Tanh,
        )
        self.fusion_net = MLP(
            [2 * element_dim, hidden_dim, output_dim],
            activation=nn.Tanh,
        )

    def forward(self, configuration: torch.Tensor) -> torch.Tensor:
        if configuration.ndim != 3 or configuration.shape[-1] != 3:
            raise ValueError("configuration must have shape (batch, num_res, 3).")
        features = physics_aware_resonator_features(
            configuration, harmonics=self.modal_harmonics
        )
        h = self.element_net(features)
        pooled = torch.cat((h.mean(dim=1), h.max(dim=1).values), dim=-1)
        return self.fusion_net(pooled)


class ResonanceQueryEncoder(nn.Module):
    """Permutation-invariant resonator/query interaction encoder.

    For every requested frequency, each resonator is combined with the query
    frequency and a normalized detuning proxy ``frequency - f_t``.  A shared
    element network followed by mean/max pooling keeps the result invariant to
    resonator ordering while exposing near-resonance information explicitly.

    Note that frequency and resonator tuning frequency use the project's existing
    normalization rules, so the detuning is dimensionless rather than Hz.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        element_dim: int = 64,
        output_dim: int = 64,
        modal_harmonics: int = 4,
    ) -> None:
        super().__init__()
        self.modal_harmonics = int(modal_harmonics)
        resonator_dim = 3 + 2 * self.modal_harmonics + self.modal_harmonics**2
        interaction_dim = resonator_dim + 4  # f, delta, |delta|, delta^2
        self.element_net = MLP(
            [interaction_dim, hidden_dim, hidden_dim, element_dim],
            activation=nn.Tanh,
        )
        self.fusion_net = MLP(
            [2 * element_dim, hidden_dim, output_dim],
            activation=nn.Tanh,
        )

    def forward(
        self,
        configuration: torch.Tensor,
        frequency: torch.Tensor,
    ) -> torch.Tensor:
        if configuration.ndim != 3 or configuration.shape[-1] != 3:
            raise ValueError("configuration must have shape (batch, num_res, 3).")
        if frequency.ndim != 3 or frequency.shape[-1] != 1:
            raise ValueError("frequency must have shape (batch, n_freq, 1).")
        if configuration.shape[0] != frequency.shape[0]:
            raise ValueError("configuration and frequency batch sizes must match.")

        b, n, _ = configuration.shape
        f = frequency.shape[1]
        resonator = physics_aware_resonator_features(
            configuration, harmonics=self.modal_harmonics
        )
        resonator = resonator[:, None, :, :].expand(b, f, n, -1)

        query = frequency[:, :, None, :].expand(b, f, n, 1)
        f_t = configuration[:, None, :, 0:1].expand(b, f, n, 1)
        delta = query - f_t
        pair = torch.cat((resonator, query, delta, delta.abs(), delta.square()), dim=-1)
        h = self.element_net(pair)
        pooled = torch.cat((h.mean(dim=2), h.max(dim=2).values), dim=-1)
        return self.fusion_net(pooled)


# ==================================================
# Dataset compatibility / full-spectrum loaders
# ==================================================


def _configuration_features(dataset) -> np.ndarray:
    """Read current configuration features, with legacy name support."""
    if hasattr(dataset, "configuration_features"):
        value = dataset.configuration_features
    elif hasattr(dataset, "design_features"):
        value = dataset.design_features
    else:
        raise AttributeError("ERP dataset has no configuration/design feature array.")
    return np.asarray(value, dtype=np.float32)


def _split_ids(dataset) -> dict[str, np.ndarray]:
    """Read current configuration split IDs, with legacy name support."""
    if hasattr(dataset, "split_configuration_ids"):
        value = dataset.split_configuration_ids
    elif hasattr(dataset, "split_design_ids"):
        value = dataset.split_design_ids
    else:
        value = None
    if value is None:
        raise RuntimeError("ERP dataset has not been split.")
    return {name: np.asarray(ids, dtype=np.int64) for name, ids in value.items()}


class ERPSpectrumDataset(Dataset):
    """One item = one complete resonator configuration and ERP spectrum."""

    def __init__(self, dataset, configuration_ids: Sequence[int]) -> None:
        if dataset.norm_params is None:
            raise RuntimeError("Dataset normalization must be available first.")

        ids = np.asarray(configuration_ids, dtype=np.int64)
        norm = dataset.norm_params

        configuration = normalize_configuration_array(
            _configuration_features(dataset)[ids], norm
        )
        frequency = normalize_frequency_array(dataset.frequency_values, norm)
        response = normalize_erp_array(
            np.asarray(dataset.responses, dtype=np.float32)[ids, :, 0], norm
        )

        self.configuration = torch.from_numpy(configuration.astype(np.float32))
        self.frequency = torch.from_numpy(frequency[:, None].astype(np.float32))
        self.response = torch.from_numpy(response[..., None].astype(np.float32))

    def __len__(self) -> int:
        return self.configuration.shape[0]

    def __getitem__(self, index: int):
        return self.configuration[index], self.frequency, self.response[index]


def build_spectrum_loaders(
    dataset,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    seed: int = 727,
) -> dict[str, DataLoader]:
    """Create configuration-level loaders for full-spectrum operator training."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    splits = _split_ids(dataset)
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(seed)

    loaders: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        subset = ERPSpectrumDataset(dataset, splits[name])
        loaders[name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            generator=generator if name == "train" else None,
        )
    return loaders


def prepare_operator_data(
    num_configurations: int = 500,
    *,
    batch_size: int = 16,
    dataset_file: str = DEFAULT_DATASET_FILE,
    regenerate_dataset: bool = False,
    num_generate: int | None = None,
    num_res: int = default_num_res,
    seed: int = 727,
    preprocessing_state: Mapping[str, object] | None = None,
    num_workers: int = 0,
    verbose: bool = True,
):
    """Prepare the common ERP data and configuration-level spectrum loaders."""
    dataset, _ = prepare_erp_dataset(
        num_samples=num_configurations,
        batch_size=max(64, batch_size),
        num_res=num_res,
        dataset_file=dataset_file,
        regenerate_dataset=regenerate_dataset,
        num_generate=num_generate,
        seed=seed,
        preprocessing_state=preprocessing_state,
        num_workers=0,
        verbose=verbose,
    )
    loaders = build_spectrum_loaders(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    return dataset, loaders


# ==================================================
# Loss, training and evaluation
# ==================================================


def erp_spectrum_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    slope_weight: float = 0.5,
) -> torch.Tensor:
    """Normalized ERP MSE plus a small first-difference penalty."""
    mse = F.mse_loss(prediction, target)
    if slope_weight <= 0.0 or prediction.shape[1] < 2:
        return mse
    dp = prediction[:, 1:] - prediction[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return mse + float(slope_weight) * F.mse_loss(dp, dt)


def _full_batch_tensors(
    loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate every batch a loader yields into one full-batch tensor triple.

    L-BFGS needs a fixed objective across its internal line-search
    evaluations within one ``step()`` call; a shuffled mini-batch loader
    would hand it a different loss function each time it re-evaluates the
    closure, which breaks its curvature estimate. Using the loader's own
    collation (rather than reaching into dataset internals) keeps this
    consistent with how ``train``/``val`` batches are built everywhere else.
    """
    configurations, frequencies, targets = [], [], []
    for configuration, frequency, target in loader:
        configurations.append(configuration)
        frequencies.append(frequency)
        targets.append(target)
    return (
        torch.cat(configurations, dim=0).to(device, dtype=torch.float32),
        torch.cat(frequencies, dim=0).to(device, dtype=torch.float32),
        torch.cat(targets, dim=0).to(device, dtype=torch.float32),
    )


def train_operator(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    *,
    epochs: int = 200,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    slope_weight: float = 0.5,
    lbfgs_epochs: int = 0,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 10,
    plot: bool = True,
    save_plots: bool = True,
    operator_name: str = "operator",
    plots_dir: str | Path | None = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train a common-API neural operator and retain best-validation weights.

    ``lbfgs_epochs`` (default 0, disabled) appends a full-batch L-BFGS
    fine-tuning phase after the ordinary AdamW+cosine-schedule loop -- the
    standard "Adam then L-BFGS polish" recipe from physics-informed/
    scientific-ML training. Each of the ``lbfgs_epochs`` outer steps calls
    ``optimizer.step(closure)`` once (internally up to ``lbfgs_max_iter``
    strong-Wolfe line-search iterations against the *entire* training set,
    not a mini-batch), and both phases share the same best-validation-loss
    checkpointing, so the final restored weights are whichever phase
    actually reached the lower validation loss.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if lbfgs_epochs < 0:
        raise ValueError("lbfgs_epochs cannot be negative.")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    history = {"train": [], "val": []}
    best_val = math.inf
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_sum = 0.0
        train_count = 0

        for configuration, frequency, target in loaders["train"]:
            configuration = configuration.to(device, dtype=torch.float32, non_blocking=True)
            frequency = frequency.to(device, dtype=torch.float32, non_blocking=True)
            target = target.to(device, dtype=torch.float32, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(configuration, frequency)
            loss = erp_spectrum_loss(prediction, target, slope_weight=slope_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch = configuration.shape[0]
            train_sum += loss.item() * batch
            train_count += batch

        train_loss = train_sum / max(train_count, 1)
        history["train"].append(train_loss)

        model.eval()
        val_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for configuration, frequency, target in loaders["val"]:
                configuration = configuration.to(device, dtype=torch.float32, non_blocking=True)
                frequency = frequency.to(device, dtype=torch.float32, non_blocking=True)
                target = target.to(device, dtype=torch.float32, non_blocking=True)
                prediction = model(configuration, frequency)
                loss = erp_spectrum_loss(prediction, target, slope_weight=slope_weight)
                batch = configuration.shape[0]
                val_sum += loss.item() * batch
                val_count += batch

        val_loss = val_sum / max(val_count, 1)
        history["val"].append(val_loss)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch + 1:4d}/{epochs} | "
            f"train={train_loss:.6e} | val={val_loss:.6e}"
        )

    if lbfgs_epochs > 0:
        train_configuration, train_frequency, train_target = _full_batch_tensors(
            loaders["train"]
        )
        val_configuration, val_frequency, val_target = _full_batch_tensors(loaders["val"])

        lbfgs_optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs_optimizer.zero_grad(set_to_none=True)
            prediction = model(train_configuration, train_frequency)
            loss = erp_spectrum_loss(prediction, train_target, slope_weight=slope_weight)
            loss.backward()
            return loss

        model.train()
        for step in range(lbfgs_epochs):
            train_loss = float(lbfgs_optimizer.step(closure).detach())
            history["train"].append(train_loss)

            model.eval()
            with torch.inference_mode():
                val_prediction = model(val_configuration, val_frequency)
                val_loss = float(
                    erp_spectrum_loss(val_prediction, val_target, slope_weight=slope_weight)
                )
            model.train()
            history["val"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"L-BFGS step {step + 1:4d}/{lbfgs_epochs} | "
                f"train={train_loss:.6e} | val={val_loss:.6e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_plots or plot:
        plot_dir = operator_plot_dir(operator_name, plots_dir) if save_plots else None
        plot_loss_curves(
            history["train"],
            history["val"],
            title=f"{operator_name} ERP training loss",
            ylabel="Normalized loss",
            log_y=True,
            save_path=(plot_dir / "loss_curve.png") if plot_dir else None,
            show=plot,
        )

    return model, history


def evaluate_operator(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    norm_params: Mapping[str, object],
    frequency_values: np.ndarray,
    *,
    plot: bool = True,
    num_plot: int = 5,
    save_plots: bool = True,
    operator_name: str = "operator",
    plots_dir: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate complete test spectra in physical ERP units.

    Saves a full-test parity plot and up to ``num_plot`` complete ERP spectrum
    comparisons when ``save_plots=True``. ``plot`` controls display only.
    """
    model = model.to(device)
    model.eval()
    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    configuration_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for configuration, frequency, target in loaders["test"]:
            configuration_batches.append(configuration.numpy())
            configuration = configuration.to(device, dtype=torch.float32, non_blocking=True)
            frequency = frequency.to(device, dtype=torch.float32, non_blocking=True)
            prediction = model(configuration, frequency)
            pred_batches.append(prediction.cpu().numpy())
            true_batches.append(target.numpy())

    pred_norm = np.concatenate(pred_batches, axis=0)[..., 0]
    true_norm = np.concatenate(true_batches, axis=0)[..., 0]
    pred = denormalize_erp_array(pred_norm, norm_params)
    true = denormalize_erp_array(true_norm, norm_params)
    configurations = denormalize_configuration_array(
        np.concatenate(configuration_batches, axis=0), norm_params
    )

    error = pred - true
    mse = float(np.mean(error**2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    mean_error = float(np.mean(error))
    error_std = float(np.std(error))

    # Global agreement statistics across every test-spectrum frequency point.
    true_flat = true.reshape(-1).astype(np.float64, copy=False)
    pred_flat = pred.reshape(-1).astype(np.float64, copy=False)
    true_centered = true_flat - np.mean(true_flat)
    pred_centered = pred_flat - np.mean(pred_flat)
    correlation_denominator = float(
        np.sqrt(np.sum(true_centered**2) * np.sum(pred_centered**2))
    )
    pearson_correlation = (
        float(np.sum(true_centered * pred_centered) / correlation_denominator)
        if correlation_denominator > 0.0
        else float("nan")
    )

    r2_denominator = float(np.sum(true_centered**2))
    r2_score = (
        float(1.0 - np.sum((pred_flat - true_flat) ** 2) / r2_denominator)
        if r2_denominator > 0.0
        else float("nan")
    )

    # Also summarize Pearson correlation spectrum-by-spectrum. This captures
    # whether each individual predicted ERP curve follows the ground-truth
    # spectral shape, rather than only measuring agreement after flattening.
    per_spectrum_pearson: list[float] = []
    for true_spectrum, pred_spectrum in zip(true, pred, strict=True):
        true_spectrum = np.asarray(true_spectrum, dtype=np.float64)
        pred_spectrum = np.asarray(pred_spectrum, dtype=np.float64)
        true_spectrum_centered = true_spectrum - np.mean(true_spectrum)
        pred_spectrum_centered = pred_spectrum - np.mean(pred_spectrum)
        denominator = float(
            np.sqrt(
                np.sum(true_spectrum_centered**2)
                * np.sum(pred_spectrum_centered**2)
            )
        )
        if denominator > 0.0:
            per_spectrum_pearson.append(
                float(
                    np.sum(true_spectrum_centered * pred_spectrum_centered)
                    / denominator
                )
            )

    mean_spectrum_pearson = (
        float(np.mean(per_spectrum_pearson))
        if per_spectrum_pearson
        else float("nan")
    )

    freq = np.asarray(frequency_values, dtype=np.float64)
    true_peak_idx = np.argmax(true, axis=1)
    pred_peak_idx = np.argmax(pred, axis=1)
    peak_frequency_mae = float(
        np.mean(np.abs(freq[pred_peak_idx] - freq[true_peak_idx]))
    )
    true_peak_amp = true[np.arange(true.shape[0]), true_peak_idx]
    pred_at_true_peak = pred[np.arange(pred.shape[0]), true_peak_idx]
    peak_amplitude_mae = float(np.mean(np.abs(pred_at_true_peak - true_peak_amp)))

    print("=" * 68)
    print(f"ERP test RMSE               : {rmse:.6f} dB")
    print(f"ERP test MAE                : {mae:.6f} dB")
    print(f"ERP mean error (bias)       : {mean_error:.6f} dB")
    print(f"ERP error standard deviation: {error_std:.6f} dB")
    print(f"Pearson correlation         : {pearson_correlation:.6f}")
    print(f"Mean spectrum Pearson corr. : {mean_spectrum_pearson:.6f}")
    print(f"R^2 score                   : {r2_score:.6f}")
    print(f"Dominant peak frequency MAE : {peak_frequency_mae:.4f} Hz")
    print(f"ERP error at true peak MAE  : {peak_amplitude_mae:.6f} dB")
    print("=" * 68)

    plot_dir = operator_plot_dir(operator_name, plots_dir) if save_plots else None
    n_to_plot = min(max(int(num_plot), 0), true.shape[0])

    if save_plots or plot:
        for i in range(n_to_plot):
            plot_erp_comparison(
                freq,
                true[i],
                pred[i],
                title=f"{operator_name} - test configuration {i + 1}",
                configuration=configurations[i],
                save_path=(
                    plot_dir / f"erp_spectrum_test_config_{i + 1:02d}.png"
                    if plot_dir
                    else None
                ),
                show=plot,
            )

        plot_prediction_scatter(
            true,
            pred,
            xlabel="Ground Truth ERP (dB)",
            ylabel="Predicted ERP (dB)",
            title=f"{operator_name} - test ERP prediction vs ground truth",
            save_path=(plot_dir / "prediction_vs_ground_truth.png") if plot_dir else None,
            show=plot,
        )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mean_error_db": mean_error,
        "error_std_db": error_std,
        "pearson_correlation": pearson_correlation,
        "mean_spectrum_pearson_correlation": mean_spectrum_pearson,
        "r2_score": r2_score,
        "peak_frequency_mae_hz": peak_frequency_mae,
        "peak_amplitude_mae_db": peak_amplitude_mae,
        "predictions": pred,
        "targets": true,
        "configurations": configurations,
        "plot_directory": str(plot_dir) if plot_dir is not None else None,
    }


# ==================================================
# Prediction and checkpoint helpers
# ==================================================


def predict_erp_spectrum(
    model: nn.Module,
    configuration: np.ndarray,
    norm_params: Mapping[str, object],
    *,
    frequency_values: np.ndarray = freqs,
    compare_solver: bool = True,
    plot: bool = True,
    save_plots: bool = True,
    operator_name: str = "operator",
    plots_dir: str | Path | None = None,
) -> dict[str, object]:
    """Predict ERP for one raw [f_t,x,y] resonator configuration."""
    configuration = np.asarray(configuration, dtype=np.float32)
    normalized_configuration = normalize_configuration_array(configuration, norm_params)

    frequency_values = np.asarray(frequency_values, dtype=np.float32)
    normalized_frequency = normalize_frequency_array(frequency_values, norm_params)

    config_tensor = torch.from_numpy(normalized_configuration).unsqueeze(0).to(device)
    freq_tensor = torch.from_numpy(normalized_frequency[:, None]).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        pred_norm = model(config_tensor, freq_tensor).cpu().numpy()[0, :, 0]
    prediction = denormalize_erp_array(pred_norm, norm_params)

    result: dict[str, object] = {
        "configuration": configuration,
        "frequencies": frequency_values,
        "prediction": prediction,
    }

    ground_truth = None
    if compare_solver:
        ground_truth = compute_erp_spectrum(
            configuration_to_resonators(configuration),
            frequencies=frequency_values,
        )
        result["ground_truth"] = ground_truth

    if save_plots or plot:
        plot_dir = operator_plot_dir(operator_name, plots_dir) if save_plots else None
        if ground_truth is not None:
            plot_erp_comparison(
                frequency_values,
                ground_truth,
                prediction,
                title=f"{operator_name} - ERP spectrum prediction",
                configuration=configuration,
                save_path=(plot_dir / "prediction_spectrum.png") if plot_dir else None,
                show=plot,
            )
        else:
            # Use the same comparison plot API; when no solver curve exists,
            # no plot is created because a meaningful comparison is unavailable.
            print("Prediction computed; no ground truth requested for comparison plot.")
        result["plot_directory"] = str(plot_dir) if plot_dir is not None else None

    return result


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_operator_checkpoint(
    model: nn.Module,
    *,
    operator_name: str,
    model_config: Mapping[str, object],
    preprocessing_state: Mapping[str, object],
    filename: str,
) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "operator_name": operator_name,
            "model_config": dict(model_config),
            "model_state_dict": model.state_dict(),
            "preprocessing_state": preprocessing_state,
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def load_operator_checkpoint(filename: str) -> dict[str, object]:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train the operator first.")
    return torch.load(path, map_location=device, weights_only=False)


# ==================================================
# Common experiment workflow
# ==================================================


def run_operator_experiment(
    *,
    operator_name: str,
    build_model: Callable[..., nn.Module],
    model_config: Mapping[str, object],
    action: str = "train",
    num_configurations: int = 500,
    batch_size: int = 16,
    epochs: int = 200,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    slope_weight: float = 0.5,
    lbfgs_epochs: int = 0,
    lbfgs_max_iter: int = 20,
    lbfgs_history_size: int = 10,
    dataset_file: str = DEFAULT_DATASET_FILE,
    regenerate_dataset: bool = False,
    num_generate: int | None = None,
    seed: int = 727,
    num_workers: int = 0,
    checkpoint_file: str | None = None,
    configuration: np.ndarray | None = None,
    plot: bool = True,
    save_plots: bool = True,
    plots_dir: str | Path | None = None,
    num_evaluation_plots: int = 5,
    evaluate_after_training: bool = False,
) -> dict[str, object]:
    """Train, evaluate, or predict with one operator architecture."""
    action = str(action).lower().strip()
    if action not in {"train", "evaluate", "predict"}:
        raise ValueError("action must be 'train', 'evaluate', or 'predict'.")

    seed_everything(seed)
    if checkpoint_file is None:
        checkpoint_file = f"models/{operator_name.lower()}_erp.pth"

    if action == "train":
        dataset, loaders = prepare_operator_data(
            num_configurations=num_configurations,
            batch_size=batch_size,
            dataset_file=dataset_file,
            regenerate_dataset=regenerate_dataset,
            num_generate=num_generate,
            seed=seed,
            num_workers=num_workers,
            verbose=True,
        )
        model = build_model(num_res=dataset.num_res, **dict(model_config)).to(device)
        print(f"{operator_name} trainable parameters: {parameter_count(model):,}")
        model, history = train_operator(
            model,
            loaders,
            epochs=epochs,
            lr=learning_rate,
            weight_decay=weight_decay,
            slope_weight=slope_weight,
            lbfgs_epochs=lbfgs_epochs,
            lbfgs_max_iter=lbfgs_max_iter,
            lbfgs_history_size=lbfgs_history_size,
            plot=plot,
            save_plots=save_plots,
            operator_name=operator_name,
            plots_dir=plots_dir,
        )
        save_operator_checkpoint(
            model,
            operator_name=operator_name,
            model_config=model_config,
            preprocessing_state=dataset.preprocessing_state(),
            filename=checkpoint_file,
        )

        result: dict[str, object] = {
            "action": action,
            "operator_name": operator_name,
            "model": model,
            "dataset": dataset,
            "loaders": loaders,
            "history": history,
        }
        if evaluate_after_training:
            result["metrics"] = evaluate_operator(
                model,
                loaders,
                dataset.norm_params,
                dataset.frequency_values,
                plot=plot,
                num_plot=num_evaluation_plots,
                save_plots=save_plots,
                operator_name=operator_name,
                plots_dir=plots_dir,
            )
        return result

    checkpoint = load_operator_checkpoint(checkpoint_file)
    saved_config = dict(checkpoint["model_config"])
    preprocessing_state = checkpoint["preprocessing_state"]
    selected_ids = np.asarray(preprocessing_state["selected_source_ids"], dtype=np.int64)

    dataset, loaders = prepare_operator_data(
        num_configurations=int(selected_ids.size),
        batch_size=batch_size,
        dataset_file=dataset_file,
        regenerate_dataset=False,
        seed=seed,
        preprocessing_state=preprocessing_state,
        num_workers=num_workers,
        verbose=True,
    )
    model = build_model(num_res=dataset.num_res, **saved_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if action == "evaluate":
        metrics = evaluate_operator(
            model,
            loaders,
            dataset.norm_params,
            dataset.frequency_values,
            plot=plot,
            num_plot=num_evaluation_plots,
            save_plots=save_plots,
            operator_name=operator_name,
            plots_dir=plots_dir,
        )
        return {
            "action": action,
            "operator_name": operator_name,
            "model": model,
            "metrics": metrics,
            "frequency_values": np.asarray(dataset.frequency_values).copy(),
        }

    if configuration is None:
        test_id = int(_split_ids(dataset)["test"][0])
        configuration = _configuration_features(dataset)[test_id].copy()
        print("No configuration supplied; using the first test configuration:")
        print(configuration)

    prediction = predict_erp_spectrum(
        model,
        configuration,
        dataset.norm_params,
        frequency_values=dataset.frequency_values,
        compare_solver=True,
        plot=plot,
        save_plots=save_plots,
        operator_name=operator_name,
        plots_dir=plots_dir,
    )
    return {
        "action": action,
        "operator_name": operator_name,
        "model": model,
        "prediction": prediction,
    }


def make_operator_runner(
    *,
    operator_name: str,
    build_model: Callable[..., nn.Module],
    model_config: Mapping[str, object],
    default_epochs: int = 200,
    default_learning_rate: float = 5e-4,
) -> Callable[..., dict[str, object]]:
    """Create the repeated per-architecture ``main`` experiment wrapper once."""

    def runner(
        action: str = "train",
        num_configurations: int = 500,
        batch_size: int = 16,
        epochs: int = default_epochs,
        learning_rate: float = default_learning_rate,
        lbfgs_epochs: int = 0,
        dataset_file: str = DEFAULT_DATASET_FILE,
        regenerate_dataset: bool = False,
        seed: int = 727,
        configuration=None,
        plot: bool = True,
        save_plots: bool = True,
        plots_dir: str | Path | None = None,
        num_evaluation_plots: int = 5,
        evaluate_after_training: bool = False,
        checkpoint_file: str | None = None,
    ) -> dict[str, object]:
        return run_operator_experiment(
            operator_name=operator_name,
            build_model=build_model,
            model_config=model_config,
            action=action,
            num_configurations=num_configurations,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            lbfgs_epochs=lbfgs_epochs,
            dataset_file=dataset_file,
            regenerate_dataset=regenerate_dataset,
            seed=seed,
            configuration=configuration,
            checkpoint_file=checkpoint_file,
            plot=plot,
            save_plots=save_plots,
            plots_dir=plots_dir,
            num_evaluation_plots=num_evaluation_plots,
            evaluate_after_training=evaluate_after_training,
        )

    runner.__name__ = "main"
    runner.__doc__ = f"Run train/evaluate/predict workflow for {operator_name}."
    return runner


__all__ = [
    "ACTIVATIONS",
    "resolve_activation",
    "MLP",
    "ResidualMLPBlock",
    "FrequencyRefinement1d",
    "physics_aware_resonator_features",
    "ResonatorSetEncoder",
    "ResonanceQueryEncoder",
    "ERPSpectrumDataset",
    "build_spectrum_loaders",
    "prepare_operator_data",
    "erp_spectrum_loss",
    "train_operator",
    "evaluate_operator",
    "predict_erp_spectrum",
    "parameter_count",
    "save_operator_checkpoint",
    "load_operator_checkpoint",
    "run_operator_experiment",
    "make_operator_runner",
]
