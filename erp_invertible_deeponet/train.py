"""Train the invertible DeepONet variants on the 2-resonator dataset.

Variants (same architecture, different number of basis functions Q):

* ``Q8``  -- strict Kaltenbach et al.: Q = D = 8 (2 resonators x [m, f_t, x, y]).
* ``Q64`` -- padded: the design is padded with 56 latent numbers (z = 0 in the
  forward direction), so the trunk has Q = 64 basis functions.
* ``Q128`` -- padded to Q = 128 (same trunk).
* ``Q64-FNO`` / ``Q64-DCO`` -- Q = 64 with an FNO-style / DCO-style trunk
  (see model.py); the inverse is unchanged.

Training reuses the project's interruption-safe loop
(``erp_inverse_operators.train_all.train_one``: Adam + cosine LR, grad clip,
best-validation weights, resume every 5 epochs). Loss = forward MSE on the
normalised ERP + inverse-consistency term (Huber) whose weight ramps up over
the first ``INVERSE_WARMUP`` epochs, so the basis first learns the spectra.
After training, the noise variance of the coefficient posterior is set to the
training-set forward residual.

Usage (from the repository root)::

    python -m erp_invertible_deeponet.train              # both variants
    python -m erp_invertible_deeponet.train Q64 --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from erp_inverse_operators.common import prepare_inverse_data
from erp_inverse_operators.train_all import train_one
from utils.erp_dataset import DATASETS, select_dataset_modal_resolution
from utils.plotting import plot_loss_curves
from utils.support import device, seed_everything

from .model import InvertibleDeepONet

PACKAGE_ROOT = Path(__file__).resolve().parent
DATASET = "200k_2res_18modes"
VARIANTS = {"Q8": 0, "Q64": 56, "Q128": 120, "Q64-FNO": 56, "Q64-DCO": 56}  # name -> latent padding (Q = 8 + pad)
TRUNKS = {"Q64-FNO": "fno", "Q64-DCO": "dco"}  # name -> trunk architecture (default: Fourier features + MLP)
INVERSE_WARMUP = 10
LR = 5e-4


def model_path(name: str, dataset_tag: str = DATASET) -> Path:
    return PACKAGE_ROOT / "models" / dataset_tag / f"idon_{name.lower()}.pth"


def plot_dir(dataset_tag: str = DATASET, name: str = "ALL_MODELS") -> Path:
    path = PACKAGE_ROOT / "plots" / dataset_tag / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare(dataset_tag: str = DATASET, batch_size: int = 128, seed: int = 727):
    """Loaders of (normalised ERP (301,), design (2, 4) in bounded logit coordinates, sorted by f_t)."""
    select_dataset_modal_resolution(dataset_tag)
    spec = DATASETS[dataset_tag]
    files = list(spec["files"])
    return prepare_inverse_data(num_configurations=int(spec["num_configurations"]), batch_size=batch_size,
                                dataset_file=files if len(files) > 1 else files[0], seed=seed, design_param="bounded12")


def build(name: str, dataset) -> InvertibleDeepONet:
    design_dim = 4 * int(dataset.num_res)
    return InvertibleDeepONet(design_dim, n_freq=len(dataset.frequency_values), pad=VARIANTS[name],
                              trunk=TRUNKS.get(name, "mlp"))


def loss_fn(model, spectrum, design, epoch):
    weight = min(1.0, (epoch + 1) / INVERSE_WARMUP)
    return model.training_loss(spectrum, design, inverse_weight=weight)["total"]


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


def train_variant(name: str, dataset, loaders, dataset_tag: str = DATASET, epochs: int = 150, seed: int = 727):
    print(f"\n{'#' * 70}\nInvertible DeepONet {name}: D = {4 * dataset.num_res}, Q = {4 * dataset.num_res + VARIANTS[name]}"
          f" basis functions, dataset '{dataset_tag}'\n{'#' * 70}")
    model = build(name, dataset)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    path = model_path(name, dataset_tag)
    resume = path.with_suffix(".resume.pt")
    history = train_one(model, loaders, loss_fn, epochs=epochs, lr=LR, name=f"iDON-{name}", resume_path=resume)
    s2 = set_noise_variance(model, loaders["train"])
    print(f"[iDON-{name}] coefficient-posterior noise variance (normalised ERP) = {s2:.4f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "norm_params": dict(dataset.norm_params), "variant": name,
                "design_dim": 4 * int(dataset.num_res), "pad": VARIANTS[name], "trunk": TRUNKS.get(name, "mlp"), "n_freq": len(dataset.frequency_values),
                "num_res": int(dataset.num_res), "history": history, "dataset_tag": dataset_tag,
                "training_config": {"epochs": epochs, "lr": LR, "inverse_warmup_epochs": INVERSE_WARMUP,
                                    "loss": "forward MSE + w * Huber(inverse) + 0.1 * Huber(latent)", "seed": seed}}, path)
    resume.unlink(missing_ok=True)
    print(f"Saved {path}")
    plot_loss_curves(history["train"], history["val"], title=f"Invertible DeepONet {name}: training history",
                     ylabel="Loss (forward MSE + inverse consistency)", log_y=True,
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
    args = parser.parse_args()
    seed_everything(727)
    dataset, loaders = prepare(batch_size=args.batch_size)
    for variant in args.variants:
        if model_path(variant).exists():
            print(f"{model_path(variant)} exists -- skipping {variant}.")
            continue
        train_variant(variant, dataset, loaders, epochs=args.epochs)
    print("ALL DONE", flush=True)
