"""Train the invertible DeepONet variants on a registered ERP dataset
(default: the 2-resonator 200k dataset; any dataset works, e.g. ``--dataset 100k``
with 3 resonators).

The design is D = 4 x num_res numbers ([m, f_t, x, y] per resonator, k derived),
so D = 8 for 2 resonators and D = 12 for 3. Variants (same architecture,
different number of basis functions Q):

* ``Q8``  -- strict Kaltenbach et al.: Q = D (8 for 2 resonators, 12 for 3).
* ``Q64`` -- padded: the design is padded with Q - D latent numbers (z = 0 in
  the forward direction), so the trunk has Q = 64 basis functions.
* ``Q128`` -- padded to Q = 128 (same trunk).
* ``Q64-FNO`` / ``Q64-DCO`` / ``Q64-DNO`` / ``Q64-WNO`` / ``Q64-LNO`` /
  ``Q64-SIREN`` -- Q = 64 with a trunk built from that forward architecture's
  layers (see model.py); the inverse is unchanged.

Training reuses the project's interruption-safe loop
(``erp_inverse.scripts.train_all.train_one``: Adam + cosine LR, grad clip,
best-validation weights, resume every 5 epochs). Loss = forward MSE on the
normalised ERP + inverse-consistency term (Huber) whose weight ramps up over
the first ``INVERSE_WARMUP`` epochs, so the basis first learns the spectra.
After training, the noise variance of the coefficient posterior is set to the
training-set forward residual.

Usage (from the repository root)::

    python -m erp_invertible_deeponet.scripts.train                       # every variant, 200k_2res_18modes
    python -m erp_invertible_deeponet.scripts.train Q64 --epochs 100
    python -m erp_invertible_deeponet.scripts.train Q64 Q128 --dataset 100k   # 3 resonators
    python -m erp_invertible_deeponet.scripts.train Q64-ERP                    # Q64 trained with the ERP loss
    (Q64-ERP uses MSE + slope + peak term from 80 % of the epochs, every other variant plain MSE;
     --loss erp/mse overrides, --retrain retrains variants that already have a checkpoint)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from erp_inverse.scripts.common import prepare_inverse_data
from erp_inverse.scripts.train_all import train_one
from utils.erp_dataset import DATASETS, select_dataset_modal_resolution
from utils.plotting import plot_loss_curves
from utils.support import device, seed_everything

from .model import InvertibleDeepONet

from utils.paths import IDON_ROOT as PACKAGE_ROOT, idon_model_path, idon_plot_dir

DATASET = "200k_2res_18modes"
# name -> number of basis functions Q ("D": strict Q = D, no padding)
VARIANTS = {"Q8": "D", "Q64": 64, "Q128": 128, "Q64-FNO": 64, "Q64-DCO": 64, "Q64-DNO": 64, "Q64-WNO": 64,
            "Q64-LNO": 64, "Q64-SIREN": 64, "Q64-ERP": 64}
# name -> trunk architecture (default: Fourier features + MLP, the DeepONet trunk)
TRUNKS = {"Q64-FNO": "fno", "Q64-DCO": "dco", "Q64-DNO": "dno", "Q64-WNO": "wno", "Q64-LNO": "lno",
          "Q64-SIREN": "siren"}
# name -> forward loss ("mse": plain MSE; "erp": MSE + slope + peak term, see below)
LOSSES = {"Q64-ERP": "erp"}
INVERSE_WARMUP = 10
LR = 5e-4
# Forward ERP loss, as for the forward operators: MSE + SLOPE_WEIGHT * MSE of the
# first difference along frequency + PEAK_WEIGHT * squared error at the true
# spectrum's peaks; the peak term is switched on after PEAK_START_FRACTION of the
# epochs (the basis first learns the overall spectra, then sharpens the peaks).
# Validation always uses the final loss (peak term on), so every epoch is scored
# with the same objective.  loss="mse": plain MSE (the original iDON loss, every
# variant except those listed in LOSSES).
SLOPE_WEIGHT = 0.5
PEAK_WEIGHT = 0.05
PEAK_START_FRACTION = 0.8


def model_path(name: str, dataset_tag: str = DATASET) -> Path:
    """erp_invertible_deeponet/models/<dataset>/idon_<variant>.pth"""
    return idon_model_path(name, dataset_tag)


def plot_dir(dataset_tag: str = DATASET, name: str = "ALL_MODELS") -> Path:
    """erp_invertible_deeponet/plots/models/<dataset>/<variant>/"""
    return idon_plot_dir(dataset_tag, name)


def prepare(dataset_tag: str = DATASET, batch_size: int = 128, seed: int = 727):
    """Loaders of (normalised ERP (301,), design (2, 4) in bounded logit coordinates, sorted by f_t)."""
    select_dataset_modal_resolution(dataset_tag)
    spec = DATASETS[dataset_tag]
    files = list(spec["files"])
    return prepare_inverse_data(num_configurations=int(spec["num_configurations"]), batch_size=batch_size,
                                dataset_file=files if len(files) > 1 else files[0], seed=seed, design_param="bounded12")


def padding(name: str, design_dim: int) -> int:
    """Latent padding Q - D of a variant for a design of ``design_dim`` numbers."""
    q = VARIANTS[name]
    if q == "D":
        return 0
    if int(q) < design_dim:
        raise ValueError(f"{name}: Q = {q} is smaller than the design dimension D = {design_dim}.")
    return int(q) - int(design_dim)


def build(name: str, dataset) -> InvertibleDeepONet:
    design_dim = 4 * int(dataset.num_res)
    return InvertibleDeepONet(design_dim, n_freq=len(dataset.frequency_values), pad=padding(name, design_dim),
                              trunk=TRUNKS.get(name, "mlp"))


def make_loss_fn(epochs: int, loss: str = "erp"):
    """Training loss of epoch ``epoch`` (0-based); see SLOPE_WEIGHT / PEAK_WEIGHT."""
    peak_start = int(round(PEAK_START_FRACTION * epochs))

    def loss_fn(model, spectrum, design, epoch):
        weight = min(1.0, (epoch + 1) / INVERSE_WARMUP)
        slope = SLOPE_WEIGHT if loss == "erp" else 0.0
        peak = PEAK_WEIGHT if loss == "erp" and epoch >= peak_start else 0.0
        return model.training_loss(spectrum, design, inverse_weight=weight, slope_weight=slope,
                                   peak_weight=peak)["total"]

    return loss_fn


def loss_description(epochs: int, loss: str = "erp") -> str:
    if loss != "erp":
        return "forward MSE + w * Huber(inverse) + 0.1 * Huber(latent)"
    return (f"forward [MSE + {SLOPE_WEIGHT} * slope MSE + {PEAK_WEIGHT} * peak SE (from epoch "
            f"{int(round(PEAK_START_FRACTION * epochs)) + 1})] + w * Huber(inverse) + 0.1 * Huber(latent)")


@torch.no_grad()
def set_noise_variance(model, loader, max_batches: int = 200) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, (spectrum, design) in enumerate(loader):
        if i >= max_batches:
            break
        spectrum, design = spectrum.to(device), design.to(device)
        pred = model(design.reshape(design.shape[0], -1))
        total += ((pred - spectrum) ** 2).sum().item()
        n += spectrum.numel()
    model.noise_var.fill_(total / n)
    return total / n


def train_variant(name: str, dataset, loaders, dataset_tag: str = DATASET, epochs: int = 150, seed: int = 727,
                  early_stop_epoch: int | None = 50, loss: str | None = None):
    """Train one variant (``loss``: None = the variant's own, see LOSSES); stops at ``early_stop_epoch`` if the validation loss has
    stopped improving (None: always run all epochs)."""
    print(f"\n{'#' * 70}\nInvertible DeepONet {name}: D = {4 * dataset.num_res}, Q = {4 * dataset.num_res + padding(name, 4 * dataset.num_res)}"
          f" basis functions, dataset '{dataset_tag}'\n{'#' * 70}")
    model = build(name, dataset)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    path = model_path(name, dataset_tag)
    resume = path.with_suffix(".resume.pt")
    loss = loss or LOSSES.get(name, "mse")
    print(f"Loss: {loss_description(epochs, loss)}")
    history = train_one(model, loaders, make_loss_fn(epochs, loss), epochs=epochs, lr=LR, name=f"iDON-{name}", resume_path=resume,
                        early_stop_epoch=early_stop_epoch)
    stopped = history.pop("stopped_early_at", None)
    s2 = set_noise_variance(model, loaders["train"])
    print(f"[iDON-{name}] coefficient-posterior noise variance (normalised ERP) = {s2:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "norm_params": dict(dataset.norm_params), "variant": name,
                "design_dim": 4 * int(dataset.num_res), "pad": padding(name, 4 * int(dataset.num_res)), "trunk": TRUNKS.get(name, "mlp"), "n_freq": len(dataset.frequency_values),
                "num_res": int(dataset.num_res), "history": history, "dataset_tag": dataset_tag,
                "training_config": {"epochs": epochs, "epochs_run": len(history["val"]), "stopped_early": stopped is not None,
                                    "lr": LR, "inverse_warmup_epochs": INVERSE_WARMUP,
                                    "loss": loss_description(epochs, loss), "loss_mode": loss, "seed": seed}}, path)
    resume.unlink(missing_ok=True)
    print(f"Saved {path}")
    plot_loss_curves(history["train"], history["val"], title=f"Invertible DeepONet {name}: training history",
                     ylabel="Loss (forward + inverse consistency)", log_y=True,
                     save_path=plot_dir(dataset_tag, name) / "loss_curve.png", show=False)
    return model


def load_variant(name: str, dataset_tag: str = DATASET):
    ckpt = torch.load(model_path(name, dataset_tag), map_location=device, weights_only=False)
    model = InvertibleDeepONet(ckpt["design_dim"], n_freq=ckpt["n_freq"], pad=ckpt["pad"], trunk=ckpt.get("trunk", "mlp"))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt["norm_params"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset", default=DATASET, choices=sorted(DATASETS), help="registered dataset tag")
    parser.add_argument("--no-early-stop", action="store_true", help="always train all epochs (default: stop at "
                        "epoch 50 if the validation loss stopped improving)")
    parser.add_argument("--loss", choices=("erp", "mse"), default=None,
                        help="override the variant's loss -- erp: MSE + slope + peak term from 80%% of the epochs; "
                             "mse: plain MSE (default: Q64-ERP erp, every other variant mse)")
    parser.add_argument("--retrain", action="store_true", help="retrain variants that already have a checkpoint")
    args = parser.parse_args()
    seed_everything(727)
    dataset, loaders = prepare(args.dataset, batch_size=args.batch_size)
    for variant in args.variants:
        if model_path(variant, args.dataset).exists() and not args.retrain:
            print(f"{model_path(variant, args.dataset)} exists -- skipping {variant} (--retrain to redo).")
            continue
        train_variant(variant, dataset, loaders, dataset_tag=args.dataset, epochs=args.epochs,
                      early_stop_epoch=None if args.no_early_stop else 50, loss=args.loss)
    print("ALL DONE", flush=True)
