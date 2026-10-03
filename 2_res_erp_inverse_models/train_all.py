"""Train the position-only inverse models (MDN, Flow, Diffusion) and evaluate each one.

Training strategy (identical to ``erp_inverse_operators.train_all``, whose
training loop is reused): Adam + cosine LR annealing, gradient clipping at 5,
best-validation-epoch weights, fixed validation randomness, non-finite batches
skipped, and the full training state saved every 5 epochs (``<model>.resume.pt``)
so an interrupted run continues where it stopped. MDN component means are
warm-started by k-means++ on real training designs.

After each model: checkpoint ``models/<dataset>/<model>.pth``, loss curve, and
the full solver-scored evaluation (``evaluate.py``) in ``plots/<dataset>/<MODEL>/``.
At the end: the all-model comparison in ``plots/<dataset>/ALL_MODELS/``.

Usage (from the repository root)::

    python -m 2_res_erp_inverse_models.train_all                      # all three, 100 epochs
    python -m 2_res_erp_inverse_models.train_all --models MDN --epochs 50
"""

from __future__ import annotations

import argparse

import torch

from erp_inverse_operators.train_all import _plot_all_histories, _warm_start_if_mdn, train_one
from utils.plotting import plot_loss_curves
from utils.support import seed_everything

from .common import DEFAULT_DATASET, model_path, plot_dir, prepare_data, save_checkpoint
from .registry import INVERSE_MODELS, SHORT_TO_KEY


def train_model(key: str, dataset, loaders, dataset_tag: str, *, epochs: int, batch_size: int, seed: int,
                spectrum_encoder: str = "pooled"):
    spec = INVERSE_MODELS[key]
    name = spec["short"]
    norm = dataset.norm_params
    print("\n" + "#" * 70 + f"\nTraining {spec['name']} ({name}) on '{dataset_tag}': "
          f"{2 * dataset.num_res} positions from the {norm['f_low']:g}-{norm['f_high']:g} Hz ERP\n" + "#" * 70)
    model = spec["build"](num_res=dataset.num_res, spectrum_encoder=spectrum_encoder)
    _warm_start_if_mdn(model, loaders)
    path = model_path(name, dataset_tag)
    resume_path = path.with_suffix(".resume.pt")
    history = train_one(model, loaders, spec["loss_fn"], epochs=epochs, lr=spec["lr"], name=name,
                        resume_path=resume_path)
    save_checkpoint(model, norm, path, extra={
        "model_short": name, "num_res": int(dataset.num_res), "spectrum_encoder": spectrum_encoder,
        "history": history, "dataset_tag": dataset_tag,
        "training_config": {
            "optimizer": "Adam", "learning_rate": float(spec["lr"]), "epochs": int(epochs),
            "batch_size": int(batch_size), "lr_schedule": "CosineAnnealingLR(eta_min=0.01*lr)",
            "grad_clip_norm": 5.0, "seed": int(seed),
            "design": "positions [x, y] per resonator, logit-bounded to the plate, sorted by x",
            "erp_band_hz": [float(norm["f_low"]), float(norm["f_high"])],
        },
    })
    resume_path.unlink(missing_ok=True)
    plot_loss_curves(history["train"], history["val"], title=f"{name}: training history ({dataset_tag})",
                     ylabel="Loss (NLL for MDN / Flow, noise MSE for Diffusion)", log_y=False,
                     save_path=plot_dir(dataset_tag, name) / "loss_curve.png", show=False)
    return model, history


def main(dataset_tag: str = DEFAULT_DATASET, models: list[str] | None = None, epochs: int | None = None,
         batch_size: int = 64, seed: int = 727, spectrum_encoder: str = "pooled", skip_existing: bool = True,
         evaluate_models: bool = True, num_test_examples: int = 500, num_samples: int = 16):
    from . import evaluate

    seed_everything(seed)
    dataset, loaders = prepare_data(dataset_tag, batch_size=batch_size, seed=seed)
    names = models or [spec["short"] for spec in INVERSE_MODELS.values()]
    histories = {}
    for name in names:
        key = SHORT_TO_KEY[name]
        path = model_path(name, dataset_tag)
        if skip_existing and path.exists():
            print(f"{path} exists -- skipping training of {name}.")
            histories[name] = torch.load(path, map_location="cpu", weights_only=False).get("history", {"train": [], "val": []})
        else:
            model_epochs = int(epochs) if epochs is not None else int(INVERSE_MODELS[key]["epochs"])
            _, histories[name] = train_model(key, dataset, loaders, dataset_tag, epochs=model_epochs,
                                             batch_size=batch_size, seed=seed, spectrum_encoder=spectrum_encoder)
        if evaluate_models:
            evaluate.evaluate_model(name, dataset_tag, data=(dataset, loaders),
                                    num_test_examples=num_test_examples, num_samples=num_samples)
        print(f"MODEL DONE {name}", flush=True)
    _plot_all_histories(histories, plot_dir(dataset_tag) / "all_models_loss.png", dataset_tag)
    if evaluate_models:
        evaluate.compare_models(dataset_tag, names)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--models", default="MDN,Flow,Diffusion", help="comma-separated: MDN, Flow, Diffusion")
    parser.add_argument("--epochs", type=int, default=None, help="default: each model's registry value (100)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--spectrum-encoder", default="pooled", choices=("pooled", "positional"))
    parser.add_argument("--retrain", action="store_true", help="retrain models that already have a checkpoint")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()
    main(args.dataset, [m.strip() for m in args.models.split(",") if m.strip()], args.epochs, args.batch_size,
         spectrum_encoder=args.spectrum_encoder, skip_existing=not args.retrain, evaluate_models=not args.no_eval)
