"""Central registry for the 7 trainable inverse models (6 probabilistic +
the deterministic, forward-surrogate-in-the-loop SurrogateInverse).

Mirrors ``erp_forward_operators/operator_registry.py``'s shape (a key -> spec dict) so
the CLI can select "just MDN" or "all inverse models" the same way it
already selects individual forward operators or "all forward operators".
"""

from __future__ import annotations

from erp_inverse_operators.basis_flow import BasisFlow
from erp_inverse_operators.cvae import ConditionalVAE
from erp_inverse_operators.diffusion import ConditionalDiffusion
from erp_inverse_operators.flow import ConditionalFlow
from erp_inverse_operators.mdn import MDN
from erp_inverse_operators.padding_inn import PadINN

NUM_RES = 3
DESIGN_DIM = NUM_RES * 5

# KL annealing schedule for the cVAE -- beta ramps from 0 to KL_TARGET_BETA
# over the first KL_WARMUP_EPOCHS epochs instead of being fixed from step 1,
# which is the classic cause of posterior collapse (see cvae.py).
KL_WARMUP_EPOCHS = 30
KL_TARGET_BETA = 0.1


def _mdn_loss(model, spectrum, design, epoch):
    return model.training_loss(spectrum, design)


def _cvae_loss(model, spectrum, design, epoch):
    beta = KL_TARGET_BETA * min(1.0, epoch / KL_WARMUP_EPOCHS)
    return model.training_loss(spectrum, design, beta=beta)


def _flow_loss(model, spectrum, design, epoch):
    return model.training_loss(spectrum, design)


def _diffusion_loss(model, spectrum, design, epoch):
    return model.training_loss(spectrum, design)


def _basis_flow_loss(model, spectrum, design, epoch):
    return model.training_loss(spectrum, design)


def _padding_inn_loss(model, spectrum, design, epoch):
    return model.training_loss(spectrum, design)


def _surrogate_loss(model, spectrum, design, epoch, *, norm_params):
    # Needs the TRAINING dataset's own normalisation to cross into each
    # frozen surrogate's physical units -- bound by train_all at train time.
    return model.training_loss(spectrum, design, own_norm_params=norm_params, surrogate_weight=1.0)


def _build_surrogate(dataset_tag: str | None = None):
    from erp_inverse_operators.surrogate_inverse import (
        DEFAULT_SURROGATE_CHECKPOINTS,
        SurrogateInverse,
        surrogate_checkpoints_for,
    )

    checkpoints = surrogate_checkpoints_for(dataset_tag) if dataset_tag else DEFAULT_SURROGATE_CHECKPOINTS
    return SurrogateInverse(design_dim=DESIGN_DIM, surrogate_checkpoints=checkpoints)


INVERSE_MODELS = {
    "1": {
        "name": "Mixture Density Network",
        "short": "MDN",
        "build": lambda: MDN(design_dim=DESIGN_DIM, num_components=10),
        "loss_fn": _mdn_loss,
        "lr": 1e-3,
        "epochs": 150,
    },
    "2": {
        "name": "Conditional VAE",
        "short": "cVAE",
        "build": lambda: ConditionalVAE(design_dim=DESIGN_DIM, latent_dim=8),
        "loss_fn": _cvae_loss,
        "lr": 1e-3,
        "epochs": 150,
    },
    "3": {
        "name": "Conditional Normalizing Flow",
        "short": "Flow",
        "build": lambda: ConditionalFlow(design_dim=DESIGN_DIM, num_layers=8, hidden=96),
        "loss_fn": _flow_loss,
        "lr": 5e-4,
        "epochs": 150,
    },
    "4": {
        "name": "Conditional Diffusion",
        "short": "Diffusion",
        "build": lambda: ConditionalDiffusion(design_dim=DESIGN_DIM, num_steps=100, hidden=128),
        "loss_fn": _diffusion_loss,
        "lr": 1e-3,
        "epochs": 150,
    },
    "5": {
        "name": "Basis Flow (invertible, bidirectional)",
        "short": "BasisFlow",
        "build": lambda: BasisFlow(design_dim=DESIGN_DIM, n_freq=301, num_layers=8, flow_hidden=96),
        "loss_fn": _basis_flow_loss,
        "lr": 5e-4,
        "epochs": 150,
    },
    "6": {
        "name": "Padding INN (Ardizzone et al., arXiv:1808.04730)",
        "short": "PadINN",
        "build": lambda: PadINN(design_dim=DESIGN_DIM, n_freq=301, z_dim=16, num_layers=8, hidden=96),
        "loss_fn": _padding_inn_loss,
        "lr": 5e-4,
        "epochs": 150,
    },
    "7": {
        "name": "Surrogate-in-the-loop inverse (deterministic, frozen DCO+GNO)",
        "short": "Surrogate",
        "build": _build_surrogate,          # optional arg: dataset tag
        "loss_fn": _surrogate_loss,         # needs norm_params=..., see train_all
        "needs_norm_params": True,
        "lr": 5e-4,
        "epochs": 100,
    },
}

SHORT_TO_KEY = {spec["short"]: key for key, spec in INVERSE_MODELS.items()}

__all__ = ["INVERSE_MODELS", "SHORT_TO_KEY", "NUM_RES", "DESIGN_DIM"]
