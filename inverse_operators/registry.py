"""Central registry for the 4 trainable probabilistic inverse models.

Mirrors ``operators/operator_registry.py``'s shape (a key -> spec dict) so
the CLI can select "just MDN" or "all inverse models" the same way it
already selects individual forward operators or "all forward operators".
"""

from __future__ import annotations

from inverse_operators.cvae import ConditionalVAE
from inverse_operators.diffusion import ConditionalDiffusion
from inverse_operators.flow import ConditionalFlow
from inverse_operators.mdn import MDN

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
}

__all__ = ["INVERSE_MODELS", "NUM_RES", "DESIGN_DIM"]
