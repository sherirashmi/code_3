"""Train the inverse-design models (all 7, or a selection) and validate them.

Validation here does NOT check "did we recover the exact original design" --
the inverse problem is genuinely non-unique (see module docstrings), so that
would be the wrong test. Instead: sample several candidate designs per
target spectrum from each model, run every sampled design through the
*actual coupled plate-resonator solver* (``utils.solver.compute_erp_spectrum``
-- the same modal solver that generated the dataset, not a neural forward
surrogate), and check how well the resulting predicted spectra match the
target. Using the real solver instead of a forward operator removes that
operator's own approximation error from the inverse-model score entirely.

Training fixes applied here (see the per-method module docstrings for why):
  - Cosine LR annealing for every model (was a flat LR for all 80 epochs).
  - Best-validation-epoch checkpointing (was: whatever the last epoch
    produced, which silently kept MDN's overfit epoch-80 weights).
  - KL annealing for the cVAE (beta ramps up over the first few epochs
    instead of being fixed from step 1, which is the classic cause of
    posterior collapse -- and this model showed exactly that signature:
    a flat loss curve and near-zero sample-to-sample diversity).
  - Cosine noise schedule + fewer steps for the diffusion model (see
    diffusion.py's docstring).

Per-sample reporting: for MDN, Flow, and cVAE -- the three methods with a
tractable ``log p(design | spectrum)`` -- every sampled design is reported
with its exact log-probability under the model, plus a softmax-normalized
"relative confidence" across that method's own samples for the same target
(these sum to 1 and are the more directly interpretable number: "how much
more plausible is this candidate than the others sampled for the same
target", not an absolute probability, which a density value alone isn't).
Diffusion has no tractable density with this simple formulation; it's
reported instead with a spectrum-consistency score (how well the sampled
design's solver-simulated spectrum matches the target) -- explicitly
labeled as such, not as a probability, so the two aren't confused.
"""

from __future__ import annotations

import copy
import functools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.erp_dataset import dataset_tag_for, denormalize_erp_array
from utils.paths import inverse_model_path, inverse_plot_dir
from utils.plotting import FREQ_LABEL, ERP_LABEL, model_colors, plot_loss_curves, save_figure

from erp_inverse_operators.common import prepare_inverse_data, save_checkpoint, denormalize_design
from erp_inverse_operators.evaluate import SPAWN_CONTEXT, score_samples, solve_configs
from erp_inverse_operators.mdn import MDN
from erp_inverse_operators.padding_inn import PadINN
from erp_inverse_operators.registry import INVERSE_MODELS, NUM_RES, DESIGN_DIM, variant_name
from utils.support import device, seed_everything


def _warm_start_if_mdn(model, loaders) -> None:
    """MDN's mixture-component means benefit from a k-means warm-start on
    real designs (see MDN.initialize_from_data's docstring) -- every other
    registry model has no equivalent hook, so this is a targeted no-op for
    them. Pulls one batch's worth of real training designs from the loader
    already in hand, so no extra data loading is needed.
    """
    if not isinstance(model, MDN):
        return
    for _spectrum, design in loaders["train"]:
        model.initialize_from_data(design)
        break


def _padinn_checkpoint_extra(model, loaders, max_calibration_examples: int = 500) -> dict | None:
    """PadINN's raw (temperature=1) samples were found to be severely
    overconfident (see padding_inn.py's calibrate_temperature docstring);
    picking a good temperature needs held-out data, so this runs once right
    after training and stores the result in the checkpoint, where
    evaluate.py's load_inverse_model picks it up and makes it every later
    caller's (predict.py/evaluate_design.py/paper_style_report.py's, via
    that same loader) default -- rather than leaving calibrate_temperature
    a correct but never-invoked method.

    Capped to ``max_calibration_examples`` validation rows: calibration draws
    ``num_samples`` (64 by default) posterior samples per row per candidate
    temperature, each requiring a full flow forward pass -- on the full
    validation split (thousands of rows for the 100k-configuration dataset)
    that's an expensive one-off pass on this project's CPU-only environment
    for no added statistical benefit (500 rows x design_dim already gives a
    coverage estimate over thousands of values).
    """
    if not isinstance(model, PadINN):
        return None
    val_spectrum, val_design = [], []
    collected = 0
    for spectrum, design in loaders["val"]:
        val_spectrum.append(spectrum)
        val_design.append(design)
        collected += spectrum.shape[0]
        if collected >= max_calibration_examples:
            break
    val_spectrum = torch.cat(val_spectrum, dim=0)[:max_calibration_examples].to(device)
    val_design = torch.cat(val_design, dim=0)[:max_calibration_examples].to(device)
    model.eval()
    temperature = model.calibrate_temperature(val_spectrum, val_design)
    print(f"[PadINN] calibrated sample() temperature = {temperature} (validation coverage check, "
          f"{val_spectrum.shape[0]} examples)")
    return {"padinn_temperature": temperature}


VALIDATION_SEED = 20251001
# Validation is scored at the END of every annealing schedule (e.g. the
# cVAE's final KL weight), so the val loss means the same thing every epoch.
_VALIDATION_EPOCH = 10**6


@torch.no_grad()
def validation_loss(model, loader, loss_fn) -> float:
    """Mean validation loss with FIXED randomness.

    cVAE / Diffusion / PadINN losses draw random latents, timesteps or noise;
    with fresh draws every epoch the "best epoch" partly reflects Monte-Carlo
    luck. Re-seeding the RNG identically for every validation pass (inside a
    forked RNG state, so training randomness is untouched) makes epochs
    directly comparable.
    """
    model.eval()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    total, n = 0.0, 0
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(VALIDATION_SEED)
        for spectrum, design in loader:
            spectrum = spectrum.to(device, non_blocking=True)
            design = design.to(device, non_blocking=True)
            loss = loss_fn(model, spectrum, design, _VALIDATION_EPOCH)
            if torch.isfinite(loss):
                total += loss.item() * spectrum.shape[0]
                n += spectrum.shape[0]
    return total / n if n else float("nan")


def train_one(model, loaders, loss_fn, epochs, lr, name):
    """Adam + cosine LR schedule, grad-clip 5, best-validation checkpointing.

    Non-finite batch losses are skipped (with a warning) rather than
    poisoning the weights; if a whole epoch is non-finite training stops
    early and the best weights so far are kept.
    """
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        total, n, skipped = 0.0, 0, 0
        for spectrum, design in loaders["train"]:
            spectrum = spectrum.to(device, non_blocking=True)
            design = design.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, spectrum, design, epoch)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total += loss.item() * spectrum.shape[0]
            n += spectrum.shape[0]
        scheduler.step()
        if n == 0:
            print(f"[{name}] epoch {epoch + 1}: every batch loss was non-finite -- stopping early.")
            break
        if skipped:
            print(f"[{name}] epoch {epoch + 1}: skipped {skipped} non-finite batch(es).")
        train_loss = total / n

        val_loss = validation_loss(model, loaders["val"], loss_fn)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
        print(f"[{name}] epoch {epoch + 1:3d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{name}] restored best-validation checkpoint (val={best_val:.4f})")
    return history


def format_configuration(configuration: np.ndarray) -> str:
    lines = []
    for i, (m, k, f_t, x, y) in enumerate(configuration):
        lines.append(f"  res{i + 1}: m={m:.3f}kg k={k:,.0f}N/m f_t={f_t:.1f}Hz x={x:.3f}m y={y:.3f}m")
    return "\n".join(lines)


def _bound_loss_fn(spec, norm_params):
    """Registry loss with the training set's norm_params bound where needed."""
    if spec.get("needs_norm_params"):
        return functools.partial(spec["loss_fn"], norm_params=norm_params)
    return spec["loss_fn"]


def resolve_variant(key: str, spectrum_encoder: str = "pooled", design_param: str = "full15") -> tuple[str, str, str]:
    """(model name, effective spectrum encoder, design_param) for a registry
    entry -- PadINN has no spectrum encoder, so it never gets the _pos suffix."""
    spec = INVERSE_MODELS[key]
    if not getattr(spec["build"], "uses_spectrum_encoder", True):
        spectrum_encoder = "pooled"
    return variant_name(spec["short"], spectrum_encoder, design_param), spectrum_encoder, design_param


def _train_and_save(
    key, dataset, loaders, *, epochs, batch_size, dataset_file, num_configurations, seed,
    spectrum_encoder="pooled", design_param="full15",
):
    """Train one registry model on already-prepared loaders, then save its
    checkpoint (``models/<dataset>/inverse_<model>.pth``) and loss curve
    (``plots/<dataset>/<MODEL>/loss_curve.png``). ``<model>`` carries the
    variant suffix (``_pos`` positional spectrum encoder, ``_b12`` bounded
    12-D design space)."""
    spec = INVERSE_MODELS[key]
    short, spectrum_encoder, design_param = resolve_variant(key, spectrum_encoder, design_param)
    tag = dataset_tag_for(dataset_file)
    norm = dataset.norm_params

    print("\n" + "#" * 70 + f"\nTraining {spec['name']} ({short}) on dataset '{tag}'\n" + "#" * 70)
    model = spec["build"](design_param=design_param, spectrum_encoder=spectrum_encoder, dataset_tag=tag)
    _warm_start_if_mdn(model, loaders)
    history = train_one(model, loaders, _bound_loss_fn(spec, norm), epochs=epochs, lr=spec["lr"], name=short)

    extra = dict(_padinn_checkpoint_extra(model, loaders) or {})
    extra.update(
        {
            "model_short": short,
            "variant": {"spectrum_encoder": spectrum_encoder, "design_param": design_param},
            "history": history,
            "dataset_file": list(dataset_file) if isinstance(dataset_file, (list, tuple)) else dataset_file,
            "dataset_tag": tag,
            "num_configurations": int(num_configurations),
            "modal_resolution": list(getattr(dataset, "modal_resolution", (15, 10))),
            "training_config": {
                "optimizer": "Adam", "learning_rate": float(spec["lr"]), "epochs": int(epochs),
                "batch_size": int(batch_size), "lr_schedule": "CosineAnnealingLR(eta_min=0.01*lr)",
                "grad_clip_norm": 5.0, "seed": int(seed),
            },
        }
    )
    save_checkpoint(model, norm, inverse_model_path(short, tag), extra=extra)
    plot_loss_curves(
        history["train"], history["val"],
        title=f"{short}: training history ({tag})",
        ylabel="Loss (model-specific NLL / MSE)", log_y=True,
        save_path=inverse_plot_dir(tag, short) / "loss_curve.png", show=False,
    )
    return model, history


def train_one_inverse_model(
    key: str,
    num_configurations: int = 10000,
    epochs: int | None = None,
    batch_size: int = 64,
    dataset_file="datasets/dataset_erp_ft.pth",
    seed: int = 727,
    spectrum_encoder: str = "pooled",
    design_param: str = "full15",
):
    """Train and checkpoint exactly one inverse model (``INVERSE_MODELS`` key).

    ``epochs`` defaults to that model's own registry default when omitted.
    """
    spec = INVERSE_MODELS[key]
    epochs = int(epochs) if epochs is not None else int(spec["epochs"])
    seed_everything(seed)
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=batch_size,
        dataset_file=dataset_file, seed=seed, design_param=design_param,
    )
    model, history = _train_and_save(
        key, dataset, loaders, epochs=epochs, batch_size=batch_size,
        dataset_file=dataset_file, num_configurations=num_configurations, seed=seed,
        spectrum_encoder=spectrum_encoder, design_param=design_param,
    )
    return model, history, dataset


def main(
    num_configurations: int = 10000,
    epochs: int | None = 150,
    batch_size: int = 64,
    dataset_file="datasets/dataset_erp_ft.pth",
    keys: list[str] | None = None,
    seed: int = 727,
    spectrum_encoder: str = "pooled",
    design_param: str = "full15",
):
    """Train every (or the selected) inverse model on ONE shared dataset/split,
    save each one's checkpoint + loss curve, the all-models loss figure and
    the solver-scored validation figure + per-sample report."""
    seed_everything(seed)
    dataset, loaders = prepare_inverse_data(
        num_configurations=num_configurations, batch_size=batch_size,
        dataset_file=dataset_file, seed=seed, design_param=design_param,
    )
    norm = dataset.norm_params
    tag = dataset_tag_for(dataset_file)
    keys = list(keys) if keys else list(INVERSE_MODELS)

    models, histories = {}, {}
    for key in keys:
        spec = INVERSE_MODELS[key]
        model_epochs = int(epochs) if epochs is not None else int(spec["epochs"])
        model, history = _train_and_save(
            key, dataset, loaders, epochs=model_epochs, batch_size=batch_size,
            dataset_file=dataset_file, num_configurations=num_configurations, seed=seed,
            spectrum_encoder=spectrum_encoder, design_param=design_param,
        )
        name = resolve_variant(key, spectrum_encoder, design_param)[0]
        models[name] = (model,)
        histories[name] = history

    out_dir = inverse_plot_dir(tag)  # plots/<dataset>/ALL_MODELS
    _plot_all_histories(histories, out_dir / "all_models_loss.png", tag)

    # ------------------------------------------------------------------
    # Validation + per-sample reporting.
    # ------------------------------------------------------------------
    solver_pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT)

    num_examples = 3
    num_samples = 6
    test_spectrum, test_design = [], []
    for spectrum, design in loaders["test"]:
        test_spectrum.append(spectrum)
        test_design.append(design)
        if sum(s.shape[0] for s in test_spectrum) >= num_examples:
            break
    test_spectrum = torch.cat(test_spectrum, dim=0)[:num_examples]
    freq_hz = np.asarray(dataset.frequency_values)
    true_erp = denormalize_erp_array(test_spectrum.numpy(), norm)
    test_spectrum = test_spectrum.to(device)

    report_lines = []

    fig, axes = plt.subplots(
        num_examples, len(models), figsize=(4.8 * len(models), 4.2 * num_examples), squeeze=False
    )
    for row in range(num_examples):
        report_lines.append(f"\n{'=' * 90}\nExample {row + 1}\n{'=' * 90}")
        for col, name in enumerate(models):
            model = models[name][0]
            model.eval()
            with torch.no_grad():
                result = model.sample(test_spectrum[row : row + 1], num_samples=num_samples)
            flat_samples = (result[0] if isinstance(result, tuple) else result)
            # Physically consistent designs + the density of EXACTLY those
            # designs (re-scored after projection; see evaluate.score_samples).
            physical_all, log_p_all = score_samples(model, test_spectrum[row : row + 1], flat_samples, norm)
            physical = physical_all[0]  # (num_samples, num_res, 5)
            has_log_prob = log_p_all is not None
            log_probs = torch.from_numpy(log_p_all[0]) if has_log_prob else None
            predicted_erp = solve_configs(solver_pool, physical, freq_hz)  # (num_samples, n_freq), real dB
            recon_mse = ((predicted_erp - true_erp[row][None, :]) ** 2).mean(axis=1)

            if has_log_prob:
                log_probs_np = log_probs.numpy()
                confidence = np.exp(log_probs_np - log_probs_np.max())
                confidence = confidence / confidence.sum()
                best_idx = int(log_probs_np.argmax())
                score_label = "log p(design|spectrum); % = softmax over these samples (relative ranking, not calibrated)"
                scores = log_probs_np
            else:
                # Diffusion has no tractable density; use spectrum-consistency
                # (negative reconstruction MSE) as an explicit, differently-
                # labeled confidence proxy instead.
                confidence = np.exp(-recon_mse) / np.exp(-recon_mse).sum()
                best_idx = int(recon_mse.argmin())
                score_label = "solver spectrum-consistency (-MSE, uses the TARGET -- oracle; NOT a probability)"
                scores = -recon_mse

            report_lines.append(f"\n--- {name} ({score_label}) ---")
            for i in range(num_samples):
                marker = " <-- best" if i == best_idx else ""
                report_lines.append(
                    f"sample {i + 1}: score={scores[i]:+.4f}  relative_rank_share={confidence[i] * 100:5.1f}%"
                    f"  recon_MSE={recon_mse[i]:.4f}{marker}"
                )
                report_lines.append(format_configuration(physical[i]))

            ax = axes[row, col]
            for i in range(num_samples):
                if i == best_idx:
                    continue
                ax.plot(freq_hz, predicted_erp[i], color="#4C72B0", alpha=0.3, lw=1.1)
            ax.plot(freq_hz, predicted_erp[best_idx], color="#C44E52", lw=2.0, label="Best sample")
            ax.plot(freq_hz, true_erp[row], color="black", lw=2, label="Target")
            best_conf = confidence[best_idx] * 100
            ax.text(
                0.02, 0.98,
                f"selected: {best_conf:.0f}% of\nsoftmax (ranking)",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
            )
            if row == 0:
                ax.set_title(name, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Example {row + 1}\n{ERP_LABEL}")
            ax.set_xlabel(FREQ_LABEL)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        f"Inverse-design validation: {num_samples} sampled designs per method "
        "(best-scoring highlighted red), run through the actual solver (not a neural surrogate)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out_dir / "validation_reconstructions.png")

    report_path = out_dir / "sample_configurations_and_scores.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"Saved per-sample configurations + scores to {report_path}")

    solver_pool.shutdown()
    print("\nDONE")


def _plot_all_histories(histories, save_path, tag):
    """One panel per model (losses are model-specific NLL/MSE and NOT on a
    common scale, so overlaying them on one axis would be misleading)."""
    names = list(histories)
    if not names:
        return
    colors = model_colors(names)
    ncols = min(4, len(names))
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        tr = np.asarray(histories[name]["train"], dtype=float)
        va = np.asarray(histories[name]["val"], dtype=float)
        ax.plot(np.arange(1, tr.size + 1), tr, color=colors[name], lw=1.6, label="Training")
        ax.plot(np.arange(1, va.size + 1), va, color=colors[name], lw=1.6, ls="--", label="Validation")
        both = np.concatenate([tr, va])
        if both.size and np.nanmin(both) > 0:
            ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.grid(True, which="both")
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    fig.suptitle(f"Inverse models: training history ({tag})")
    fig.tight_layout()
    save_figure(fig, save_path)


if __name__ == "__main__":
    main()
