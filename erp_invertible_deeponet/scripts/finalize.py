"""Finalise a stopped training run from its resume file (best-validation weights).

``train_one`` saves the full state (incl. the best-validation weights) every 5
epochs to ``idon_<variant>.resume.pt``. After a run is stopped early, this turns
that state into the normal checkpoint: best weights loaded, the coefficient-
posterior noise variance set from the training data, loss curve plotted.

Usage:  python -m erp_invertible_deeponet.scripts.finalize Q8 [Q64]
"""

from __future__ import annotations

import sys

import torch

from utils.plotting import plot_loss_curves
from utils.support import device

from .train import DATASET, INVERSE_WARMUP, LR, TRUNKS, VARIANTS, build, model_path, plot_dir, prepare, set_noise_variance


def finalize(name: str, dataset, loaders) -> None:
    path = model_path(name)
    state = torch.load(path.with_suffix(".resume.pt"), map_location=device, weights_only=False)
    model = build(name, dataset).to(device)
    model.load_state_dict(state["best_state"])
    history = state["history"]
    s2 = set_noise_variance(model, loaders["train"])
    best_epoch = int(min(range(len(history["val"])), key=history["val"].__getitem__)) + 1
    print(f"[iDON-{name}] stopped after epoch {state['epoch']}: best validation loss {state['best_val']:.4f} "
          f"(epoch {best_epoch}); noise variance {s2:.4f}")
    torch.save({"model_state_dict": model.state_dict(), "norm_params": dict(dataset.norm_params), "variant": name,
                "design_dim": 4 * int(dataset.num_res), "pad": VARIANTS[name], "trunk": TRUNKS.get(name, "mlp"), "n_freq": len(dataset.frequency_values),
                "num_res": int(dataset.num_res), "history": history, "dataset_tag": DATASET,
                "training_config": {"epochs_planned": int(state["epochs"]), "epochs_run": int(state["epoch"]),
                                    "stopped_early": True, "best_epoch": best_epoch, "lr": LR,
                                    "inverse_warmup_epochs": INVERSE_WARMUP,
                                    "loss": "forward MSE + w * Huber(inverse) + 0.1 * Huber(latent)"}}, path)
    print(f"Saved {path}")
    plot_loss_curves(history["train"], history["val"], title=f"Invertible DeepONet {name}: training history "
                     f"(stopped after epoch {state['epoch']})", ylabel="Loss (forward MSE + inverse consistency)",
                     log_y=True, save_path=plot_dir(DATASET, name) / "loss_curve.png", show=False)


if __name__ == "__main__":
    dataset, loaders = prepare()
    for variant in sys.argv[1:]:
        finalize(variant, dataset, loaders)
