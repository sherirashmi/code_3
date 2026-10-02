"""Three-step training (Long et al. arXiv:2402.11722, Sec 3.3) for any
model in this package's registry, then forward + inverse evaluation on
this project's config<->ERP problem. The same schedule drives iFNO, iDCO
and iGNO.

Training schedule (hyperparameters = module constants below, all
overridable from ``main.py``):

  Stage 1 -- invertible coupling blocks + P/Q/P'/Q' (every parameter except
             the VAE), Adam lr 5e-4, cosine to 1% of lr:
                 J_IFB = J_FWD + J_INV + J_{P,Q'} + J_{P',Q}
  Stage 2 -- beta-VAE alone on TRUE designs (design-space pretraining),
             Adam lr 1e-3, KL weight beta linearly warmed 0 -> 0.05 over
             the first KL_WARMUP_EPOCHS epochs:
                 J_beta-VAE = MSE(recon, design) + beta * KL
  Stage 3 -- every parameter jointly, Adam lr 3e-4, beta = 0.05:
                 J = J_FWD + J_{P,Q'} + J_{P',Q} + J_beta-VAE + w_inv * J_INV
             (w_inv = STAGE3_INVERSE_WEIGHT = 1.0; 0.0 is the paper's eq 11)

  Every J_* term is an MSE on z-scored quantities (ERP or the f_t-sorted
  design vector). Gradient clipping at norm 5; the best-validation weights
  of each stage are restored before the next stage starts.

Evaluation:
  - Forward: predict_spectrum(configuration) vs the true ERP on the test
    split, RMSE/MAE/R^2/Pearson + the same 5 per-configuration plots and
    parity plot the forward operators produce.
  - Inverse: draw ``num_samples`` designs per held-out target spectrum,
    project them onto the physically consistent design space
    (k = m (2 pi f_t)^2, see erp_inverse_operators.common), pick one per
    target WITHOUT the solver using the model's own forward direction (the
    sample whose predicted spectrum best matches the target -- the point of
    an invertible operator), then score that design with the ACTUAL coupled
    solver. Also reports design-parameter recovery (predicted vs true).

Output layout (utils/paths.py):
    erp_invertible_operators/models/<dataset>/<model>.pth
    erp_invertible_operators/plots/<dataset>/<MODEL>/...
    erp_invertible_operators/plots/<dataset>/ALL_MODELS/...
"""

from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import torch

from erp_invertible_operators.common import InvertibleOperatorBase
from erp_invertible_operators.registry import INVERTIBLE_OPERATORS
from erp_inverse_operators.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_inverse_operators.evaluate import SPAWN_CONTEXT, solve_configs
from erp_inverse_operators.registry import NUM_RES as DEFAULT_NUM_RES
from utils.erp_dataset import (
    DATASETS,
    apply_model_modal_resolution,
    dataset_tag_for,
    denormalize_configuration_array,
    denormalize_erp_array,
    normalize_configuration_array,
)
from utils.paths import ALL_MODELS, invertible_model_path, invertible_plot_dir
from utils.plotting import (
    ERP_LABEL,
    FREQ_LABEL,
    model_colors,
    plot_all_models_loss,
    plot_prediction_scatter,
    plot_staged_loss_curves,
    save_figure,
    save_operator_experiment_plots,
)
from utils.support import device, seed_everything

STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 25
STAGE3_EPOCHS = 15
STAGE1_LR = 5e-4
STAGE2_LR = 1e-3
STAGE3_LR = 3e-4
STAGE3_INVERSE_WEIGHT = 1.0
KL_WARMUP_EPOCHS = 8
KL_TARGET_BETA = 0.05
GRAD_CLIP = 5.0
BATCH_SIZE = 128
NUM_CONFIGURATIONS = 100000
DATASET_FILE = tuple(DATASETS["100k"]["files"])
SEED = 727

PARAM_LABELS = (
    r"$m$ (kg)", r"$k$ (N/m)", r"$f_t$ (Hz)", r"$x$ (m)", r"$y$ (m)",
)


def _non_vae_parameters(model: InvertibleOperatorBase):
    return [p for name, p in model.named_parameters() if not name.startswith("vae.")]


def _run_stage(
    model: InvertibleOperatorBase,
    loaders,
    *,
    epochs: int,
    lr: float,
    parameters,
    loss_fn,
    state_module,
    tag: str,
    stage: str,
    beta_schedule=None,
    diagnose: bool = False,
):
    """Generic stage loop: Adam + cosine, grad clip, best-val restore.

    ``loss_fn(spectrum, flat_design, beta) -> dict`` with a "total" entry.
    ``state_module`` is what gets checkpointed/restored (the VAE alone in
    stage 2, the whole model otherwise).
    """
    parameters = list(parameters)
    history = {"train": [], "val": []}
    if diagnose:
        history["diagnostics"] = []
    if epochs <= 0:
        return history
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        beta = beta_schedule(epoch) if beta_schedule is not None else KL_TARGET_BETA
        model.train()
        total, n, skipped = 0.0, 0, 0
        for spectrum, design in loaders["train"]:
            spectrum = spectrum.to(device, non_blocking=True)
            flat_design = design.to(device, non_blocking=True).reshape(design.shape[0], -1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(spectrum, flat_design, beta)["total"]
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            optimizer.step()
            total += loss.item() * spectrum.shape[0]
            n += spectrum.shape[0]
        scheduler.step()
        if n == 0:
            print(f"[{tag} {stage}] every batch was non-finite in epoch {epoch + 1}; stopping this stage.")
            break
        if skipped:
            print(f"[{tag} {stage}] skipped {skipped} non-finite batch(es) in epoch {epoch + 1}.")
        train_loss = total / n

        model.eval()
        # Fixed validation randomness (VAE reparameterisation draws in
        # stages 2/3), so the best-epoch choice is not Monte-Carlo luck.
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(20251001)
            total, n = 0.0, 0
            for spectrum, design in loaders["val"]:
                spectrum = spectrum.to(device, non_blocking=True)
                flat_design = design.to(device, non_blocking=True).reshape(design.shape[0], -1)
                loss = loss_fn(spectrum, flat_design, beta)["total"]
                if torch.isfinite(loss):
                    total += loss.item() * spectrum.shape[0]
                    n += spectrum.shape[0]
            val_loss = total / n if n else float("nan")

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(state_module.state_dict())
        print(f"[{tag} {stage}] epoch {epoch + 1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | beta={beta:.4f}")
        if diagnose:
            # Numerical health of the invertible core on one validation batch.
            spectrum, design = next(iter(loaders["val"]))
            diag = model.diagnostics(
                spectrum.to(device), design.to(device).reshape(design.shape[0], -1)
            )
            history["diagnostics"].append(diag)
            print(
                f"    core: round-trip err={diag['round_trip_rel_err']:.1e} | latent norm x{diag['latent_norm_ratio']:.2f}"
                f" | gate [{diag['gate_min']:.2f}, {diag['gate_max']:.2f}] | fwd/inv latent misalignment {diag['latent_alignment']:.2f}"
            )

    if best_state is not None:
        state_module.load_state_dict(best_state)
        print(f"[{tag} {stage}] restored best-validation weights (val={best_val:.4f})")
    return history


def train_stage1(model, loaders, epochs: int, tag: str, lr: float = STAGE1_LR, cycle_weight: float = 0.0, align_weight: float = 0.0):
    return _run_stage(
        model, loaders, epochs=epochs, lr=lr, parameters=_non_vae_parameters(model),
        loss_fn=lambda s, d, _b: model.stage1_loss(s, d, cycle_weight=cycle_weight, align_weight=align_weight),
        state_module=model, tag=tag, stage="stage1", diagnose=True,
    )


def train_stage2(model, loaders, epochs: int, tag: str, lr: float = STAGE2_LR, source: str = "true"):
    """``source="true"`` (paper): VAE on true designs. ``"estimates"``: the
    VAE encoder sees the (frozen) stage-1 inverse point estimates -- the
    input it gets in stage 3 / at inference -- and decodes the true design."""
    if source == "estimates":
        def loss_fn(s, d, b):
            with torch.no_grad():
                estimate = model.infer_point_estimate(s)
            return model.stage2_loss(d, beta=b, encoder_input=estimate)
    else:
        def loss_fn(_s, d, b):
            return model.stage2_loss(d, beta=b)
    return _run_stage(
        model, loaders, epochs=epochs, lr=lr, parameters=model.vae.parameters(),
        loss_fn=loss_fn, state_module=model.vae, tag=tag, stage="stage2",
        beta_schedule=lambda epoch: KL_TARGET_BETA * min(1.0, epoch / max(KL_WARMUP_EPOCHS, 1)),
    )


def train_stage3(
    model, loaders, epochs: int, tag: str, lr: float = STAGE3_LR, inverse_weight: float = STAGE3_INVERSE_WEIGHT,
    cycle_weight: float = 0.0, align_weight: float = 0.0,
):
    return _run_stage(
        model, loaders, epochs=epochs, lr=lr, parameters=model.parameters(),
        loss_fn=lambda s, d, b: model.stage3_loss(
            s, d, beta=b, inverse_weight=inverse_weight, cycle_weight=cycle_weight, align_weight=align_weight
        ),
        state_module=model, tag=tag, stage="stage3", diagnose=True,
    )


def plot_stage_losses(tag: str, histories: dict[str, dict[str, list[float]]], save_path):
    """Three panels, one per stage (each stage optimises a different objective)."""
    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 4.2), squeeze=False)
    for ax, (stage, history) in zip(axes[0], histories.items()):
        tr = np.asarray(history["train"], dtype=float)
        va = np.asarray(history["val"], dtype=float)
        ax.plot(np.arange(1, tr.size + 1), tr, label="Training", lw=1.8)
        ax.plot(np.arange(1, va.size + 1), va, label="Validation", lw=1.8, ls="--")
        if tr.size and np.nanmin(np.concatenate([tr, va])) > 0:
            ax.set_yscale("log")
        ax.set_title(stage)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, which="both")
    fig.suptitle(f"{tag}: three-step training (Long et al., arXiv:2402.11722)")
    fig.tight_layout()
    save_figure(fig, save_path)


# ==================================================
# Evaluation
# ==================================================


def evaluate_forward(model: InvertibleOperatorBase, tag: str, dataset, loaders, norm, out_dir, num_plot: int = 5):
    model.eval()
    true_curves, pred_curves, configs_used = [], [], []
    with torch.no_grad():
        for spectrum, design in loaders["test"]:
            flat = design.to(device, non_blocking=True).reshape(design.shape[0], -1)
            pred = model.predict_spectrum_from_design(flat).squeeze(-1).cpu()  # (B, F) normalized
            true_curves.append(denormalize_erp_array(spectrum, norm))
            pred_curves.append(denormalize_erp_array(pred, norm))
            configs_used.append(design.reshape(design.shape[0], -1).numpy())
    true = np.concatenate(true_curves, axis=0)
    pred = np.concatenate(pred_curves, axis=0)
    configs_used = np.concatenate(configs_used, axis=0)

    error = pred - true
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    tc, pc = true.ravel() - true.mean(), pred.ravel() - pred.mean()
    denom = float(np.sqrt(np.sum(tc**2) * np.sum(pc**2)))
    pearson = float(np.sum(tc * pc) / denom) if denom > 0 else float("nan")
    r2_denom = float(np.sum(tc**2))
    r2 = float(1.0 - np.sum((pred.ravel() - true.ravel()) ** 2) / r2_denom) if r2_denom > 0 else float("nan")

    print("=" * 68)
    print(f"[{tag} forward] test RMSE : {rmse:.4f} dB")
    print(f"[{tag} forward] test MAE  : {mae:.4f} dB")
    print(f"[{tag} forward] Pearson    : {pearson:.4f}")
    print(f"[{tag} forward] R^2        : {r2:.4f}")
    print("=" * 68)

    # Same five spectra + parity plot the forward operators save.
    save_operator_experiment_plots(
        tag,
        metrics={
            "predictions": pred,
            "targets": true,
            "configurations": denormalize_design(configs_used, int(norm['num_res']), norm, consistent=False),
        },
        frequency_values=np.asarray(dataset.frequency_values),
        plot_dir=out_dir,
        num_configurations=num_plot,
        title_suffix=" (forward)",
    )
    return {"rmse_db": rmse, "mae_db": mae, "pearson_r": pearson, "r2": r2}


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    error = pred - true
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(np.abs(error))),
        "pearson_r": float(np.corrcoef(pred.ravel(), true.ravel())[0, 1]),
        "r2": float(1.0 - np.sum(error**2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def evaluate_inverse(
    model: InvertibleOperatorBase,
    tag: str,
    dataset,
    loaders,
    norm,
    out_dir,
    num_examples: int = 100,
    num_samples: int = 8,
    num_plot: int = 5,
):
    """Solver-scored inverse evaluation on ``num_examples`` held-out targets."""
    model.eval()
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:num_examples]
    test_design = torch.cat(test_design, dim=0)[:num_examples]
    n = test_spectrum.shape[0]
    num_plot = min(num_plot, n)
    freq_hz = np.asarray(dataset.frequency_values)
    true_erp = denormalize_erp_array(test_spectrum, norm)  # (n, F) dB
    true_design = denormalize_design(test_design.reshape(n, -1).numpy(), int(norm['num_res']), norm, consistent=False)  # (n, R, 5)

    with torch.no_grad():
        samples = model.sample(test_spectrum.to(device), num_samples=num_samples).cpu()  # (n, S, D)
        point = model.infer_point_estimate(test_spectrum.to(device)).cpu()  # (n, D)
    physical = denormalize_design(samples.numpy(), int(norm['num_res']), norm)  # (n, S, R, 5), physically consistent
    point_physical = denormalize_design(point.numpy(), int(norm['num_res']), norm)  # (n, R, 5)

    # Selection with the model's OWN forward direction: re-normalise the
    # (physically consistent) designs and pick the sample whose predicted
    # spectrum is closest to the target. This is TARGET-INFORMED (it compares
    # with the target spectrum) but solver-free -- a deployable rule for an
    # invertible model, distinct from the solver-based oracle below.
    with torch.no_grad():
        renorm = torch.from_numpy(normalize_configuration_array(physical, norm)).to(device)
        own = model.predict_spectrum(renorm.reshape(n * num_samples, int(norm['num_res']), 5)).squeeze(-1)
        own = own.reshape(n, num_samples, -1).cpu()
    own_mse = ((own - test_spectrum[:, None, :]) ** 2).mean(dim=-1).numpy()  # (n, S)
    selected = own_mse.argmin(axis=1)

    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT) as pool:
        solved = solve_configs(pool, physical.reshape(n * num_samples, int(norm['num_res']), 5), freq_hz).reshape(n, num_samples, -1)
        solved_point = solve_configs(pool, point_physical, freq_hz)  # (n, F)
    solver_mse = ((solved - true_erp[:, None, :]) ** 2).mean(axis=-1)  # (n, S)
    oracle = solver_mse.argmin(axis=1)

    rows = np.arange(n)
    pred_selected = solved[rows, selected]
    stats = {
        "num_targets": int(n),
        "num_samples": int(num_samples),
        "random_sample": _metrics(solved[:, 0], true_erp),
        "selected_by_own_forward": _metrics(pred_selected, true_erp),
        "point_estimate": _metrics(solved_point, true_erp),
        "oracle_best_of_samples": _metrics(solved[rows, oracle], true_erp),
    }
    # Predicted resonators sorted by their own f_t before slot-by-slot
    # comparison with the (f_t-sorted) truth -- predictions are not
    # constrained to stay in that order (the solver score is unaffected).
    best_design = sort_resonators_by_ft(physical[rows, selected])  # (n, R, 5)
    stats["design_recovery"] = {
        name: {
            "mae": float(np.abs(best_design[..., i] - true_design[..., i]).mean()),
            "pearson_r": float(np.corrcoef(best_design[..., i].ravel(), true_design[..., i].ravel())[0, 1]),
        }
        for i, name in enumerate(("m", "k", "f_t", "x", "y"))
    }
    print("=" * 68)
    print(f"[{tag} inverse] solver-scored, {n} targets x {num_samples} samples")
    for key, label in (("random_sample", "random sample"),
                       ("selected_by_own_forward", "own forward (target-informed)"),
                       ("point_estimate", "point estimate"),
                       ("oracle_best_of_samples", "oracle best-of-N")):
        m = stats[key]
        print(f"  {label:<24s} RMSE={m['rmse_db']:.3f} dB  MAE={m['mae_db']:.3f} dB  r={m['pearson_r']:.4f}  R^2={m['r2']:.4f}")
    print("=" * 68)

    # 1) Target vs solver-evaluated samples for the first few examples.
    fig, axes = plt.subplots(1, num_plot, figsize=(4.6 * num_plot, 4.0), squeeze=False)
    for row, ax in enumerate(axes[0]):
        for i in range(num_samples):
            if i != selected[row]:
                ax.plot(freq_hz, solved[row, i], color="#4C72B0", alpha=0.3, lw=1.0,
                        label="Other samples" if i == (1 if selected[row] == 0 else 0) else None)
        ax.plot(freq_hz, solved[row, selected[row]], color="#C44E52", lw=1.8, label="Selected sample")
        ax.plot(freq_hz, true_erp[row], color="black", lw=1.8, label="Target")
        ax.set_title(f"Example {row + 1}: MSE $= {solver_mse[row, selected[row]]:.2f}$ dB$^2$", fontsize=10)
        ax.set_xlabel(FREQ_LABEL)
        ax.grid(True)
    axes[0, 0].set_ylabel(ERP_LABEL)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{tag} inverse design: solver response of sampled designs vs target")
    fig.tight_layout()
    save_figure(fig, out_dir / "inverse_validation_reconstructions.png")

    # 2) Predicted (solver ERP of the selected design) vs true ERP.
    plot_prediction_scatter(
        true_erp, pred_selected,
        xlabel=f"Target {ERP_LABEL}", ylabel=f"Solver ERP of predicted design (dB)",
        title=f"{tag} inverse: predicted vs true ERP ({n} test targets)",
        save_path=out_dir / "inverse_prediction_vs_ground_truth.png", show=False,
    )

    # 3) Design-parameter recovery, predicted vs true.
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.3))
    for i, ax in enumerate(axes):
        t, p = true_design[..., i].ravel(), best_design[..., i].ravel()
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.scatter(t, p, s=8, alpha=0.4, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.1)
        r = stats["design_recovery"][("m", "k", "f_t", "x", "y")[i]]["pearson_r"]
        ax.set_title(f"$r = {r:.3f}$")
        ax.set_xlabel(f"True {PARAM_LABELS[i]}")
        ax.set_ylabel(f"Predicted {PARAM_LABELS[i]}")
        ax.grid(True)
    fig.suptitle(f"{tag} inverse: design-parameter recovery (non-unique problem, see erp_inverse_operators)")
    fig.tight_layout()
    save_figure(fig, out_dir / "inverse_design_parameter_recovery.png")
    return stats


# ==================================================
# Drivers
# ==================================================


# Model options (stored in the checkpoint's model_config) and training
# options (stored in its training_config). Every non-default choice adds a
# suffix to the model name, so variants never overwrite each other.
MODEL_OPTION_DEFAULTS = {
    "use_sorted_branch": False,   # _sorted : + f_t-sorted resonator branch (iFNO/iDCO)
    "design_param": "full15",     # _b12    : bounded 12-D [m, f_t, x, y], k derived
    "gate": "softplus",           # _bg     : bounded, identity-initialised gate exp(c tanh(a L))
    "readout": "pooled",          # _bin    : + ordered frequency bins in the design readout
    "spectral_padding": 0,        # _pad    : zero-padded FFT (iFNO only)
}
TRAIN_OPTION_DEFAULTS = {
    "cycle_weight": 0.0,          # _cyc    : design cycle  x -> y_hat -> x_hat
    "align_weight": 0.0,          #           + forward/inverse latent alignment
    "stage2_source": "true",      # _s2e    : stage-2 VAE pretrained on stage-1 estimates
}
RECOMMENDED_OPTIONS = {
    "design_param": "bounded12", "gate": "bounded", "readout": "binned", "spectral_padding": 8,
    "cycle_weight": 0.1, "align_weight": 0.1, "stage2_source": "estimates",
}


def variant(key: str, use_sorted_branch: bool = False, options: dict | None = None) -> tuple[str, dict, dict]:
    """(model name, model_config, training options) for a registry entry."""
    spec = INVERTIBLE_OPERATORS[key]
    opts = {**MODEL_OPTION_DEFAULTS, **TRAIN_OPTION_DEFAULTS, **dict(options or {})}
    opts["use_sorted_branch"] = bool(use_sorted_branch or opts["use_sorted_branch"])
    if not spec.get("supports_sorted_branch"):
        opts["use_sorted_branch"] = False
    if spec["short"] != "iFNO":
        opts["spectral_padding"] = 0
    name = spec["short"]
    if opts["use_sorted_branch"]:
        name += "_sorted"
    if opts["design_param"] == "bounded12":
        name += "_b12"
    if opts["gate"] == "bounded":
        name += "_bg"
    if opts["readout"] == "binned":
        name += "_bin"
    if opts["spectral_padding"]:
        name += "_pad"
    if opts["cycle_weight"] > 0 or opts["align_weight"] > 0:
        name += "_cyc"
    if opts["stage2_source"] == "estimates":
        name += "_s2e"
    model_config = dict(spec["model_config"])
    for k in ("use_sorted_branch", "design_param", "gate", "readout", "spectral_padding"):
        if opts[k] != MODEL_OPTION_DEFAULTS[k]:
            model_config[k] = opts[k]
    train_opts = {k: opts[k] for k in TRAIN_OPTION_DEFAULTS}
    return name, model_config, train_opts


def _design_dim_of(model_config: dict) -> int:
    from erp_inverse_operators.design_space import design_dim

    # num_res is stored in model_config by train_one (any resonator count);
    # checkpoints from before that default to 3.
    return design_dim(model_config.get("design_param", "full15"), int(model_config.get("num_res", DEFAULT_NUM_RES)))


def _load_checkpoint(key: str, dataset_tag: str, use_sorted_branch: bool = False, options: dict | None = None):
    spec = INVERTIBLE_OPERATORS[key]
    name, _config, _train = variant(key, use_sorted_branch, options)
    path = invertible_model_path(name, dataset_tag)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- train {name} on dataset '{dataset_tag}' first.")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if dataset_tag in DATASETS:
        # Solver checks of this model use the Nx x Ny it was trained with.
        apply_model_modal_resolution(checkpoint, DATASETS[dataset_tag]["files"], model_name=name)
    model_config = dict(checkpoint.get("model_config", spec["model_config"]))
    model = spec["build"](design_dim=_design_dim_of(model_config), **model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint


def train_one(
    key: str,
    num_configurations: int = NUM_CONFIGURATIONS,
    batch_size: int = BATCH_SIZE,
    dataset_file=DATASET_FILE,
    seed: int = SEED,
    stage1_epochs: int = STAGE1_EPOCHS,
    stage2_epochs: int = STAGE2_EPOCHS,
    stage3_epochs: int = STAGE3_EPOCHS,
    stage3_inverse_weight: float = STAGE3_INVERSE_WEIGHT,
    num_inverse_examples: int = 100,
    num_inverse_samples: int = 8,
    use_sorted_branch: bool = False,
    options: dict | None = None,
    _data=None,
):
    """Train + evaluate exactly one registry entry (e.g. key="2" for iDCO).

    ``options``: see MODEL_OPTION_DEFAULTS / TRAIN_OPTION_DEFAULTS
    (``RECOMMENDED_OPTIONS`` = all improvements switched on).
    """
    spec = INVERTIBLE_OPERATORS[key]
    tag, model_config, train_opts = variant(key, use_sorted_branch, options)
    design_param = model_config.get("design_param", "full15")
    dataset_tag = dataset_tag_for(dataset_file)

    seed_everything(seed)
    if _data is None or getattr(_data[0], "design_param", "full15") != design_param:
        _data = prepare_inverse_data(
            num_configurations=num_configurations, batch_size=batch_size, dataset_file=dataset_file,
            seed=seed, design_param=design_param,
        )
    dataset, loaders = _data
    norm = dataset.norm_params
    out_dir = invertible_plot_dir(dataset_tag, tag)
    model_config = {**model_config, "num_res": int(dataset.num_res)}  # resonators per configuration

    model: InvertibleOperatorBase = spec["build"](design_dim=_design_dim_of(model_config), **model_config).to(device)
    model.set_design_normalization(norm)
    print(f"{tag} params: {sum(p.numel() for p in model.parameters()):,} | dataset '{dataset_tag}' | "
          f"model options {model_config} | training options {train_opts}")

    cw, aw = float(train_opts["cycle_weight"]), float(train_opts["align_weight"])
    print("\n" + "#" * 70 + f"\n{tag} Stage 1: invertible coupling blocks + P/Q/P'/Q'\n" + "#" * 70)
    history1 = train_stage1(model, loaders, epochs=stage1_epochs, tag=tag, cycle_weight=cw, align_weight=aw)
    print("\n" + "#" * 70 + f"\n{tag} Stage 2: beta-VAE pretraining\n" + "#" * 70)
    history2 = train_stage2(model, loaders, epochs=stage2_epochs, tag=tag, source=train_opts["stage2_source"])
    print("\n" + "#" * 70 + f"\n{tag} Stage 3: joint fine-tuning\n" + "#" * 70)
    history3 = train_stage3(model, loaders, epochs=stage3_epochs, tag=tag, inverse_weight=stage3_inverse_weight,
                            cycle_weight=cw, align_weight=aw)
    histories = {"Stage 1 (invertible blocks)": history1, "Stage 2 ($\\beta$-VAE)": history2,
                 "Stage 3 (joint)": history3}

    training_config = {
        "stage1": {"epochs": stage1_epochs, "lr": STAGE1_LR,
                   "loss": "J_FWD + J_INV + J_PQ' + J_P'Q (+ w_cyc*J_cycle + w_align*J_align)"},
        "stage2": {"epochs": stage2_epochs, "lr": STAGE2_LR, "loss": "MSE recon + beta*KL",
                   "beta": KL_TARGET_BETA, "kl_warmup_epochs": KL_WARMUP_EPOCHS,
                   "encoder_input": train_opts["stage2_source"]},
        "stage3": {"epochs": stage3_epochs, "lr": STAGE3_LR, "beta": KL_TARGET_BETA,
                   "loss": "J_FWD + J_PQ' + J_P'Q + J_betaVAE + w_inv*J_INV (+ cycle/align)", "w_inv": stage3_inverse_weight},
        "cycle_weight": cw, "align_weight": aw,
        "optimizer": "Adam + CosineAnnealingLR(eta_min=0.01*lr)", "grad_clip_norm": GRAD_CLIP,
        "batch_size": batch_size, "num_configurations": num_configurations, "seed": seed,
        "dataset_tag": dataset_tag,
    }
    checkpoint_path = invertible_model_path(tag, dataset_tag)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(), "norm_params": dict(norm),
            "model_config": model_config, "training_config": training_config,
            "history": {"stage1": history1, "stage2": history2, "stage3": history3},
            "dataset_file": list(dataset_file) if isinstance(dataset_file, (list, tuple)) else dataset_file,
            "num_configurations": int(num_configurations),
            "modal_resolution": list(getattr(dataset, "modal_resolution", (15, 10))),
        },
        checkpoint_path,
    )
    print(f"Saved {checkpoint_path}")

    plot_staged_loss_curves(histories, f"{tag}: training history ({dataset_tag})",
                            save_path=out_dir / "loss_curve.png")
    plot_stage_losses(tag, histories, out_dir / "stage_losses.png")

    forward_metrics = evaluate_forward(model, tag, dataset, loaders, norm, out_dir)
    inverse_metrics = evaluate_inverse(model, tag, dataset, loaders, norm, out_dir,
                                       num_examples=num_inverse_examples, num_samples=num_inverse_samples)
    _write_metrics(out_dir, tag, forward_metrics, inverse_metrics)
    print(f"\n{tag} DONE -- plots in {out_dir}")
    return model, {"stage1": history1, "stage2": history2, "stage3": history3}, forward_metrics, inverse_metrics


def evaluate_one(
    key: str,
    dataset_file=DATASET_FILE,
    num_configurations: int | None = None,
    seed: int = SEED,
    num_inverse_examples: int = 100,
    num_inverse_samples: int = 8,
    use_sorted_branch: bool = False,
    options: dict | None = None,
):
    """Re-evaluate an existing checkpoint (no training)."""
    dataset_tag = dataset_tag_for(dataset_file)
    model, checkpoint = _load_checkpoint(key, dataset_tag, use_sorted_branch, options)
    tag = variant(key, use_sorted_branch, options)[0]
    design_param = dict(checkpoint.get("model_config", {})).get("design_param", "full15")
    num_configurations = int(num_configurations or checkpoint.get("num_configurations")
                             or DATASETS.get(dataset_tag, {}).get("num_configurations", NUM_CONFIGURATIONS))
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=BATCH_SIZE, dataset_file=dataset_file, seed=seed,
        design_param=design_param,
    )
    norm = dataset.norm_params
    from erp_inverse_operators.evaluate import _check_norm
    _check_norm(tag, checkpoint["norm_params"], norm)
    out_dir = invertible_plot_dir(dataset_tag, tag)
    if "history" in checkpoint:
        histories = dict(zip(("Stage 1 (invertible blocks)", "Stage 2 ($\\beta$-VAE)", "Stage 3 (joint)"),
                             checkpoint["history"].values()))
        plot_staged_loss_curves(histories, f"{tag}: training history ({dataset_tag})",
                                save_path=out_dir / "loss_curve.png")
    forward_metrics = evaluate_forward(model, tag, dataset, loaders, norm, out_dir)
    inverse_metrics = evaluate_inverse(model, tag, dataset, loaders, norm, out_dir,
                                       num_examples=num_inverse_examples, num_samples=num_inverse_samples)
    _write_metrics(out_dir, tag, forward_metrics, inverse_metrics)
    return model, checkpoint.get("history"), forward_metrics, inverse_metrics


def _write_metrics(out_dir, tag, forward_metrics, inverse_metrics) -> None:
    path = out_dir / "metrics.json"
    path.write_text(json.dumps({"model": tag, "forward": forward_metrics, "inverse": inverse_metrics}, indent=2))
    print(f"Saved {path}")


def _save_comparison(results: dict, dataset_tag: str) -> None:
    """ALL_MODELS: combined loss history + forward/inverse comparison bars."""
    out_dir = invertible_plot_dir(dataset_tag, ALL_MODELS)
    names = [r[0] for r in results.values()]
    histories = {}
    for name, history, _f, _i in results.values():
        if history:
            # Concatenate stage 1 and 3 (stage 2 trains only the VAE).
            histories[name] = {
                "train": list(history["stage1"]["train"]) + list(history["stage3"]["train"]),
                "val": list(history["stage1"]["val"]) + list(history["stage3"]["val"]),
            }
    if histories:
        plot_all_models_loss(histories, title=f"Invertible operators: stage 1 + stage 3 loss ({dataset_tag})",
                             ylabel="Loss", save_path=out_dir / "all_models_loss.png")

    colors = model_colors(names)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(names))
    fwd = [results[k][2]["rmse_db"] for k in results]
    bars = axes[0].bar(x, fwd, color=[colors[n] for n in names])
    axes[0].bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    axes[0].set_title("Forward: test RMSE")
    width = 0.27
    for j, (rule, label) in enumerate((("random_sample", "random sample"),
                                       ("selected_by_own_forward", "own forward (target-informed)"),
                                       ("oracle_best_of_samples", "best-of-N by solver (oracle)"))):
        vals = [results[k][3][rule]["rmse_db"] for k in results]
        axes[1].bar(x + (j - 1) * width, vals, width, label=label)
    axes[1].set_title("Inverse: solver-scored RMSE")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, names, rotation=15)
        ax.set_ylabel("RMSE (dB)")
        ax.grid(True, axis="y")
    fig.suptitle(f"Invertible operators ({dataset_tag})")
    fig.tight_layout()
    save_figure(fig, out_dir / "forward_inverse_rmse_bars.png")
    (out_dir / "metrics.json").write_text(json.dumps(
        {r[0]: {"forward": r[2], "inverse": r[3]} for r in results.values()}, indent=2))


def main(keys: tuple[str, ...] = ("1", "2", "3"), evaluate_only: bool = False, **kwargs):
    """Train (or only evaluate) the selected registry entries on ONE shared
    dataset/split; with more than one model, also write ALL_MODELS plots."""
    dataset_file = kwargs.get("dataset_file", DATASET_FILE)
    dataset_tag = dataset_tag_for(dataset_file)
    results = {}
    shared = None
    for key in keys:
        if evaluate_only:
            eval_kwargs = {k: v for k, v in kwargs.items()
                           if k in ("dataset_file", "num_configurations", "seed", "num_inverse_examples",
                                    "num_inverse_samples", "use_sorted_branch", "options")}
            _model, history, fwd, inv = evaluate_one(key, **eval_kwargs)
        else:
            if shared is None:
                design_param = {**MODEL_OPTION_DEFAULTS, **dict(kwargs.get("options") or {})}["design_param"]
                shared = prepare_inverse_data(
                    num_configurations=kwargs.get("num_configurations", NUM_CONFIGURATIONS),
                    batch_size=kwargs.get("batch_size", BATCH_SIZE),
                    dataset_file=dataset_file, seed=kwargs.get("seed", SEED), design_param=design_param,
                )
            _model, history, fwd, inv = train_one(key, _data=shared, **kwargs)
        results[key] = (variant(key, kwargs.get("use_sorted_branch", False), kwargs.get("options"))[0], history, fwd, inv)
    if len(results) > 1:
        _save_comparison(results, dataset_tag)
    return results


if __name__ == "__main__":
    main()
