"""Shared infrastructure for the displacement-field forward operators.

Unlike forward_operators/ (which maps design -> already spatially-integrated
ERP), every architecture in this folder maps
    (configuration, frequency, position=(x, y)) -> (displacement_real, displacement_imag)
i.e. the actual complex plate displacement field w(x, y, omega) at one
spatial point and every training frequency, trained against the
dataset_field_displacement*.pth shards (utils/field_dataset.py).

Velocity is NOT a learned output. v = i*omega*w, so given predicted
(w_real, w_imag) and the known (input) omega, velocity is an EXACT
deterministic function of what the network already predicts:
    v = i*omega*(w_real + i*w_imag) = -omega*w_imag + i*omega*w_real
    => v_real = -omega * w_imag      (imag, negated, scaled by omega)
    => v_imag =  omega * w_real      (real, scaled by omega, no sign flip)
Real and imaginary parts swap, one of the two swapped assignments picks up
a minus sign, both get scaled by omega -- exactly the relationship asked
for, confirmed here (see also compute_velocity_from_displacement below and
its docstring for the derivation, and displacement_forward_operators'
package-level docstring / the commit message for a worked numeric check).
Forcing the network to output 4 independently-learned numbers instead
would let it produce answers that don't actually satisfy v=i*omega*w --
strictly worse than building the exact relationship into the output layer
for free, so every architecture here predicts 2 numbers (w_real, w_imag),
not 4.

Architecture adaptation strategy (see this folder's other files and the
docstrings there for the per-architecture detail): every one of the 10
original forward_operators architectures keeps its distinctive
frequency-axis mechanism (DON's branch/trunk product, FNO's spectral
convolution, GNO's message passing, STO's cross-attention, WNO's wavelet
blocks, SIREN's sine layers, LNO's Laplace pole/residue expansion, DNO/DCO/
NN's residual or plain MLP stacks) COMPLETELY UNCHANGED, because frequency
is still the same shared, ordered 301-point grid every configuration uses
-- FNO's rfft and WNO's Haar transform are only mathematically valid along
an ordered, regularly-spaced axis, which frequency still is. What changes
is only the CONTEXT/CONDITIONING signal: instead of conditioning purely on
the resonator configuration, every architecture now conditions on
`FieldContextEncoder(configuration, x, y)`, which fuses the existing
ResonatorSetEncoder output with a position encoding built from the SAME
sin(m*pi*x/Lx)*sin(n*pi*y/Ly) modal basis functions the true displacement
field is expanded in (see utils/physics.py's phi_mn/mode_shapes -- this is
literally that basis, reimplemented in PyTorch so it stays differentiable
w.r.t. x, y for the physics-residual loss in physics_loss.py). Folding
position into the flat "trunk axis" instead (naively bumping every
frequency-only "1" to "3") would have silently broken FNO's/WNO's spectral/
wavelet mechanism, since collocation positions are scattered, not gridded.
"""

from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from forward_operators.neural_operator_utils import (
    MLP,
    ResonatorSetEncoder,
    physics_aware_resonator_features,
    parameter_count,
)
from utils.erp_dataset import (
    configuration_to_resonators,
    normalize_configuration_array,
    normalize_frequency_array,
)
from utils.physics import Lx, Ly
from utils.plotting import plot_erp_comparison, plot_prediction_scatter
from utils.solver import compute_erp, compute_erp_spectrum
from utils.support import device, seed_everything

OUT_DIR = Path("plots/DISPLACEMENT_FORWARD_OPERATORS")
_CONFIG_ZSCORE_FIELDS = ("m", "k", "f_t", "x", "y")


# ==================================================
# Velocity (derived, not learned)
# ==================================================


def compute_velocity_from_displacement(
    w_real: torch.Tensor, w_imag: torch.Tensor, omega: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """v = i*omega*w, exactly -- see this module's docstring for the derivation.

    ``omega`` must already be angular frequency (2*pi*f) in the same
    broadcastable shape as w_real/w_imag.
    """
    v_real = -omega * w_imag
    v_imag = omega * w_real
    return v_real, v_imag


# ==================================================
# Position features: the true modal basis, differentiable
# ==================================================


class TorchModeShapeFeatures(nn.Module):
    """Differentiable (w.r.t. x, y) reimplementation of utils.physics.mode_shapes,
    truncated to modes_x * modes_y terms (the solver itself uses the full
    Nx * Ny = 15 * 10 = 150; truncating keeps this cheap since it's evaluated
    at every collocation point in every batch, not once per configuration).

    This is not an arbitrary featurization choice: sin(m*pi*x/Lx) *
    sin(n*pi*y/Ly) is literally the basis the true displacement field is
    expanded in (w(x,y,t) = sum_mn phi_mn(x,y) * eta_mn(t), see
    utils/physics.py's phi_mn/modal_table), so handing the network these as
    input features is telling it, up front, which spatial functions the
    answer must be built from -- the same idea a Fourier/spectral operator
    uses, specialized to this plate's own known boundary conditions.

    Implemented in torch (not numpy, unlike utils/physics.py's own
    mode_shapes) specifically so it stays part of the autograd graph:
    physics_loss.py's biharmonic residual needs d^4(w)/dx^4 etc., which
    requires every step from raw x, y through to the network's output to be
    a differentiable torch op.
    """

    def __init__(self, modes_x: int = 8, modes_y: int = 6) -> None:
        super().__init__()
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.register_buffer("m_idx", torch.arange(1, self.modes_x + 1, dtype=torch.float32))
        self.register_buffer("n_idx", torch.arange(1, self.modes_y + 1, dtype=torch.float32))

    @property
    def output_dim(self) -> int:
        return self.modes_x * self.modes_y

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x, y: (..., 1) raw physical meters. Returns (..., modes_x*modes_y)."""
        x_terms = torch.sin(self.m_idx * math.pi * x / Lx)  # (..., modes_x)
        y_terms = torch.sin(self.n_idx * math.pi * y / Ly)  # (..., modes_y)
        products = x_terms[..., :, None] * y_terms[..., None, :]  # (..., modes_x, modes_y)
        return products.reshape(*products.shape[:-2], self.modes_x * self.modes_y)


class FieldContextEncoder(nn.Module):
    """The resonator-configuration context (unchanged ResonatorSetEncoder),
    fused with the query-position's modal-basis encoding. This is the ONE
    thing every architecture in this folder swaps in place of
    forward_operators' plain ``ResonatorSetEncoder(configuration)`` call --
    see this module's docstring for why this is the surgical, per-
    architecture-faithful way to add position-dependence.
    """

    def __init__(
        self,
        output_dim: int,
        resonator_hidden: int = 128,
        modes_x: int = 8,
        modes_y: int = 6,
        position_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.resonator_encoder = ResonatorSetEncoder(
            hidden_dim=resonator_hidden, element_dim=resonator_hidden, output_dim=output_dim
        )
        self.position_features = TorchModeShapeFeatures(modes_x, modes_y)
        self.position_encoder = MLP(
            [self.position_features.output_dim, position_hidden, output_dim], activation=nn.SiLU
        )
        self.fusion = MLP([2 * output_dim, output_dim, output_dim], activation=nn.SiLU)

    def forward(self, configuration: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        resonator_context = self.resonator_encoder(configuration)  # (B, output_dim)
        position_context = self.position_encoder(self.position_features(x, y))  # (B, output_dim)
        return self.fusion(torch.cat((resonator_context, position_context), dim=-1))


class FieldResonanceQueryEncoder(nn.Module):
    """Like forward_operators' ResonanceQueryEncoder (permutation-invariant,
    frequency-detuning-aware over resonators), extended with a spatial
    feature: distance from the query position to EACH resonator's own
    (x, y) attachment point -- a resonator's influence on the field is
    strongest right at its own location, exactly parallel to how detuning
    (frequency - f_t) captures "near resonance" on the frequency axis.
    Frequency-side logic is otherwise identical/unchanged.
    """

    def __init__(
        self, hidden_dim: int = 64, element_dim: int = 64, output_dim: int = 64, modal_harmonics: int = 10
    ) -> None:
        super().__init__()
        self.modal_harmonics = int(modal_harmonics)
        augmented_dim = 5 + 2 * self.modal_harmonics + self.modal_harmonics**2
        pair_dim = augmented_dim + 4 + 1  # + freq, delta, |delta|, delta^2, spatial_distance
        self.element_net = MLP([pair_dim, hidden_dim, hidden_dim, element_dim], activation=nn.Tanh)
        self.fusion_net = MLP([2 * element_dim, hidden_dim, output_dim], activation=nn.Tanh)

    def forward(
        self, configuration: torch.Tensor, frequency: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        b, n, _ = configuration.shape
        f = frequency.shape[1]
        resonator = physics_aware_resonator_features(configuration, harmonics=self.modal_harmonics)
        resonator = resonator[:, None, :, :].expand(b, f, n, -1)
        query_freq = frequency[:, :, None, :].expand(b, f, n, 1)
        f_t = configuration[:, None, :, 2:3].expand(b, f, n, 1)
        delta = query_freq - f_t

        res_x = configuration[:, None, :, 3:4].expand(b, f, n, 1)
        res_y = configuration[:, None, :, 4:5].expand(b, f, n, 1)
        qx = x[:, None, None, :].expand(b, f, n, 1)
        qy = y[:, None, None, :].expand(b, f, n, 1)
        spatial_distance = torch.sqrt((qx - res_x) ** 2 + (qy - res_y) ** 2 + 1e-12)

        pair = torch.cat((resonator, query_freq, delta, delta.abs(), delta.square(), spatial_distance), dim=-1)
        h = self.element_net(pair)  # (B, F, N, element_dim)
        pooled = torch.cat((h.mean(dim=2), h.max(dim=2).values), dim=-1)
        return self.fusion_net(pooled)  # (B, F, output_dim)


# ==================================================
# Dataset / normalization
# ==================================================


def compute_field_norm_params(
    configuration_features: np.ndarray,
    frequency_values: np.ndarray,
    real_displacement: np.ndarray,
    imag_displacement: np.ndarray,
    train_ids: np.ndarray,
    num_res: int,
) -> dict[str, float]:
    """z-score stats for configuration/frequency (same convention as
    forward_operators/utils.erp_dataset), fit on the train split only, plus
    a single shared scale for displacement (real and imag share physical
    units, so one scale -- no mean subtraction, since displacement
    oscillates symmetrically around zero by construction).
    """
    norm_params: dict[str, float] = {"num_res": int(num_res)}
    train_config = configuration_features[train_ids]
    for i, name in enumerate(_CONFIG_ZSCORE_FIELDS):
        values = train_config[..., i]
        norm_params[f"{name}_mean"] = float(values.mean())
        norm_params[f"{name}_std"] = float(values.std() + 1e-8)
    norm_params["freq_mean"] = float(frequency_values.mean())
    norm_params["freq_std"] = float(frequency_values.std() + 1e-8)

    train_real = real_displacement[train_ids]
    train_imag = imag_displacement[train_ids]
    combined = np.concatenate([train_real.ravel(), train_imag.ravel()])
    norm_params["displacement_std"] = float(combined.std() + 1e-12)
    return norm_params


class DisplacementFieldDataset(Dataset):
    """One item = one (configuration, one collocation point), all 301
    frequencies at once. Configuration is repeated across every one of its
    own collocation points -- same idea as forward_operators'
    ERPSpectrumDataset repeating one configuration across its shared
    frequency grid.

    Position (x, y) is stored RAW (physical meters), never normalized --
    the model normalizes it internally via TorchModeShapeFeatures'
    sin(m*pi*x/Lx) (a differentiable, fixed-scale operation), so autograd
    through raw x, y stays intact end-to-end for physics_loss.py's spatial
    derivatives. Configuration and frequency ARE pre-normalized here as
    usual, since the physics loss never needs gradients w.r.t. them.
    """

    def __init__(
        self,
        configuration_ids: np.ndarray,
        configuration_features: np.ndarray,
        frequency_values: np.ndarray,
        collocation_points: np.ndarray,
        real_displacement: np.ndarray,
        imag_displacement: np.ndarray,
        norm_params: Mapping[str, object],
    ) -> None:
        self.norm_params = norm_params
        points_per_config = collocation_points.shape[1]

        config_sel = configuration_features[configuration_ids]
        colloc_sel = collocation_points[configuration_ids]
        real_sel = real_displacement[configuration_ids]
        imag_sel = imag_displacement[configuration_ids]

        config_norm = normalize_configuration_array(config_sel, norm_params)
        config_flat = np.repeat(config_norm, points_per_config, axis=0)  # (C*P, N, 5)
        position_flat = colloc_sel.reshape(-1, 2).astype(np.float32)  # (C*P, 2), RAW meters

        disp_scale = float(norm_params["displacement_std"])
        real_flat = (real_sel.reshape(-1, real_sel.shape[-1]) / disp_scale).astype(np.float32)
        imag_flat = (imag_sel.reshape(-1, imag_sel.shape[-1]) / disp_scale).astype(np.float32)

        freq_norm = normalize_frequency_array(frequency_values, norm_params)

        self.configuration = torch.from_numpy(config_flat)
        self.frequency = torch.from_numpy(freq_norm.astype(np.float32))[:, None]  # (F, 1), shared
        self.position = torch.from_numpy(position_flat)  # (C*P, 2)
        self.target_real = torch.from_numpy(real_flat)  # (C*P, F)
        self.target_imag = torch.from_numpy(imag_flat)

    def __len__(self) -> int:
        return self.configuration.shape[0]

    def __getitem__(self, index: int):
        return (
            self.configuration[index],
            self.frequency,
            self.position[index, 0:1],
            self.position[index, 1:2],
            self.target_real[index][:, None],
            self.target_imag[index][:, None],
        )


def build_displacement_loaders(
    dataset_dict: Mapping[str, object],
    batch_size: int = 64,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 727,
    num_workers: int = 0,
) -> dict[str, object]:
    """Splits by CONFIGURATION (never by individual collocation point), same
    philosophy as ERPDataset's own split-by-configuration rule -- keeps a
    configuration's points from leaking across train/val/test.
    """
    configuration_features = np.asarray(dataset_dict["configuration_features"], dtype=np.float32)
    frequency_values = np.asarray(dataset_dict["frequency_values"], dtype=np.float32)
    collocation_points = np.asarray(dataset_dict["collocation_points"], dtype=np.float32)
    real_displacement = np.asarray(dataset_dict["real_displacement"], dtype=np.float32)
    imag_displacement = np.asarray(dataset_dict["imag_displacement"], dtype=np.float32)
    num_res = int(dataset_dict["num_res"])

    num_configs = configuration_features.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_configs)
    n_train = int(num_configs * train_ratio)
    n_val = int(num_configs * val_ratio)
    split_ids = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }

    norm_params = compute_field_norm_params(
        configuration_features, frequency_values, real_displacement, imag_displacement,
        split_ids["train"], num_res,
    )

    datasets = {
        name: DisplacementFieldDataset(
            ids, configuration_features, frequency_values, collocation_points,
            real_displacement, imag_displacement, norm_params,
        )
        for name, ids in split_ids.items()
    }
    loaders = {
        name: DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            num_workers=num_workers, drop_last=(name == "train" and len(ds) > batch_size),
        )
        for name, ds in datasets.items()
    }
    return {
        "loaders": loaders,
        "norm_params": norm_params,
        "split_ids": split_ids,
        "configuration_features": configuration_features,
        "frequency_values": frequency_values,
        "collocation_points": collocation_points,
        "num_res": num_res,
    }


# ==================================================
# Loss
# ==================================================


def displacement_data_loss(
    pred_real: torch.Tensor, pred_imag: torch.Tensor, target_real: torch.Tensor, target_imag: torch.Tensor
) -> torch.Tensor:
    """Plain MSE on normalized (real, imag) displacement.

    A velocity-AWARE weighting (penalizing v=i*omega*w error, not just w
    error -- higher-frequency errors get amplified by the omega factor,
    which would push training toward what ultimately matters for ERP) is a
    natural refinement but isn't implemented here: it needs its own
    consistently-normalized velocity scale (velocity and displacement do
    not share a scale once omega, which ranges over ~63-1005 rad/s, is
    multiplied in) to avoid one term silently dominating the other, which
    is a real piece of design work left for later rather than done
    half-carefully under time pressure.
    """
    return F.mse_loss(pred_real, target_real) + F.mse_loss(pred_imag, target_imag)


# ==================================================
# Training
# ==================================================


def train_displacement_operator(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    *,
    epochs: int = 100,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    physics_loss_fn=None,
    physics_weight: float = 0.05,
    operator_name: str = "operator",
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Adam + cosine schedule, best-validation checkpointing -- same recipe
    as forward_operators.train_operator. ``physics_loss_fn(model)`` (see
    physics_loss.py's ``make_physics_loss_fn``), when given, is added to
    the ordinary data loss each training step, scaled by ``physics_weight``.
    Validation is data-loss only (the physics residual isn't a held-out
    generalization signal in the usual sense -- it's a training-time
    regularizer).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    history: dict[str, list[float]] = {"train": [], "val": [], "physics": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total, total_physics, n = 0.0, 0.0, 0
        for configuration, frequency, x, y, target_real, target_imag in loaders["train"]:
            configuration = configuration.to(device)
            frequency = frequency.to(device)
            x = x.to(device)
            y = y.to(device)
            target_real = target_real.to(device)
            target_imag = target_imag.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(configuration, frequency, x, y)  # (B, F, 2)
            data_loss = displacement_data_loss(pred[..., 0:1], pred[..., 1:2], target_real, target_imag)

            loss = data_loss
            physics_value = 0.0
            if physics_loss_fn is not None:
                p_loss = physics_loss_fn(model)
                loss = loss + physics_weight * p_loss
                physics_value = float(p_loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_size = configuration.shape[0]
            total += float(data_loss.item()) * batch_size
            total_physics += physics_value * batch_size
            n += batch_size
        train_loss = total / n
        scheduler.step()

        model.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for configuration, frequency, x, y, target_real, target_imag in loaders["val"]:
                configuration = configuration.to(device)
                frequency = frequency.to(device)
                x = x.to(device)
                y = y.to(device)
                target_real = target_real.to(device)
                target_imag = target_imag.to(device)
                pred = model(configuration, frequency, x, y)
                data_loss = displacement_data_loss(pred[..., 0:1], pred[..., 1:2], target_real, target_imag)
                total += float(data_loss.item()) * configuration.shape[0]
                n += configuration.shape[0]
            val_loss = total / n

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["physics"].append(total_physics / max(1, n))
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[{operator_name}] epoch {epoch + 1:3d}/{epochs} | "
            f"train={train_loss:.5f} | val={val_loss:.5f} | physics={history['physics'][-1]:.5f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{operator_name}] restored best-validation checkpoint (val={best_val:.5f})")
    return model, history


# ==================================================
# Evaluation: reconstruct ERP from the predicted field, plot vs. true
# ==================================================


def reconstruct_erp_with_model(
    model: nn.Module,
    configuration_physical: np.ndarray,
    norm_params: Mapping[str, object],
    frequency_hz: np.ndarray,
    grid_nx: int = 40,
    grid_ny: int = 15,
) -> np.ndarray:
    """Evaluate a trained displacement operator over a spatial grid and
    integrate to ERP(dB), mirroring compute_erp_spectrum's own grid-
    integration approach (utils/solver.py) but using the model instead of
    the analytical solver. Uses a COARSER grid than the analytical
    solver's fixed 280x100 (28,000 points) deliberately: the analytical
    solver's grid is cheap (closed-form matrix algebra), but every point
    here costs a real forward pass through the network, so grid_nx*grid_ny
    trades reconstruction accuracy for compute directly -- 40*15=600 points
    is ~47x fewer evaluations than the full 280x100 grid.
    """
    from utils.physics import P_ref, c_L, rho_L
    from utils.support import grid as make_grid

    x_grid, y_grid, dA = make_grid(Lx=Lx, Ly=Ly, nx=grid_nx, ny=grid_ny)
    xs = x_grid.ravel().astype(np.float32)
    ys = y_grid.ravel().astype(np.float32)
    n_points = xs.size
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    n_freq = frequency_hz.size

    config_norm = normalize_configuration_array(configuration_physical[None], norm_params)[0]
    freq_norm = normalize_frequency_array(frequency_hz, norm_params).astype(np.float32)

    model.eval()
    with torch.no_grad():
        config_t = (
            torch.from_numpy(config_norm).to(device)[None, :, :].expand(n_points, -1, -1).contiguous()
        )
        freq_t = torch.from_numpy(freq_norm).to(device)[None, :, None].expand(n_points, -1, -1).contiguous()
        x_t = torch.from_numpy(xs).to(device)[:, None]
        y_t = torch.from_numpy(ys).to(device)[:, None]
        pred = model(config_t, freq_t, x_t, y_t)  # (P, F, 2)

    disp_scale = float(norm_params["displacement_std"])
    w_real = pred[..., 0].cpu().numpy().astype(np.float64) * disp_scale
    w_imag = pred[..., 1].cpu().numpy().astype(np.float64) * disp_scale

    omega = 2.0 * np.pi * frequency_hz[None, :]  # (1, F)
    v_real = -omega * w_imag
    v_imag = omega * w_real
    velocity = (v_real + 1j * v_imag).reshape(grid_ny, grid_nx, n_freq)

    velocity_energy = np.sum(np.abs(velocity) ** 2, axis=(0, 1)) * dA
    power = 0.5 * rho_L * c_L * velocity_energy
    return 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny) / P_ref)


def evaluate_displacement_operator(
    model: nn.Module,
    data: Mapping[str, object],
    *,
    num_plot: int = 5,
    grid_nx: int = 40,
    grid_ny: int = 15,
    save_plots: bool = True,
    plot: bool = False,
    operator_name: str = "operator",
) -> dict[str, object]:
    """Reconstructs ERP from the trained displacement field on every test
    configuration and compares against compute_erp_spectrum's true
    (analytical, full-grid) ERP -- the actual-vs-predicted ERP comparison,
    the same reporting convention forward_operators/evaluate_operator uses
    (plot_erp_comparison per configuration, plot_prediction_scatter for
    the aggregate parity plot), just fed from a field-reconstructed
    prediction instead of a directly-predicted ERP vector.
    """
    norm_params = data["norm_params"]
    frequency_values = data["frequency_values"]
    configuration_features = data["configuration_features"]
    test_ids = data["split_ids"]["test"]

    true_curves, pred_curves, configs_used = [], [], []
    for idx in test_ids:
        config_physical = configuration_features[idx]
        resonators = configuration_to_resonators(config_physical)
        true_erp = compute_erp_spectrum(resonators, frequencies=frequency_values)
        pred_erp = reconstruct_erp_with_model(
            model, config_physical, norm_params, frequency_values, grid_nx=grid_nx, grid_ny=grid_ny
        )
        true_curves.append(true_erp)
        pred_curves.append(pred_erp)
        configs_used.append(config_physical)

    true = np.stack(true_curves, axis=0)
    pred = np.stack(pred_curves, axis=0)

    error = pred - true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    true_flat, pred_flat = true.ravel(), pred.ravel()
    tc, pc = true_flat - true_flat.mean(), pred_flat - pred_flat.mean()
    denom = float(np.sqrt(np.sum(tc**2) * np.sum(pc**2)))
    pearson = float(np.sum(tc * pc) / denom) if denom > 0 else float("nan")
    r2_denom = float(np.sum(tc**2))
    r2 = float(1.0 - np.sum((pred_flat - true_flat) ** 2) / r2_denom) if r2_denom > 0 else float("nan")

    print("=" * 68)
    print(f"[{operator_name}] Field-reconstructed ERP test RMSE : {rmse:.4f} dB")
    print(f"[{operator_name}] Field-reconstructed ERP test MAE  : {mae:.4f} dB")
    print(f"[{operator_name}] Pearson correlation                : {pearson:.4f}")
    print(f"[{operator_name}] R^2                                 : {r2:.4f}")
    print("=" * 68)

    plot_dir = None
    if save_plots:
        plot_dir = OUT_DIR / operator_name.upper()
        plot_dir.mkdir(parents=True, exist_ok=True)
        n_to_plot = min(num_plot, true.shape[0])
        for i in range(n_to_plot):
            plot_erp_comparison(
                frequency_values, true[i], pred[i],
                title=f"{operator_name} (field-reconstructed) - test configuration {i + 1}",
                configuration=configs_used[i],
                save_path=plot_dir / f"erp_spectrum_test_config_{i + 1:02d}.png",
                show=plot,
            )
        plot_prediction_scatter(
            true, pred,
            xlabel="Ground Truth ERP (dB)", ylabel="Predicted ERP (dB, field-reconstructed)",
            title=f"{operator_name} - field-reconstructed ERP vs. ground truth",
            save_path=plot_dir / "prediction_vs_ground_truth.png",
            show=plot,
        )

    return {
        "rmse": rmse, "mae": mae, "pearson_correlation": pearson, "r2_score": r2,
        "true": true, "pred": pred, "plot_directory": plot_dir,
    }
