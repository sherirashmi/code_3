"""The three position-only inverse models: key -> spec (as erp_inverse_operators.registry)."""

from __future__ import annotations

from erp_inverse_operators.common import POOLED

from .design_space import design_dim
from .diffusion import ConditionalDiffusion
from .flow import ConditionalFlow
from .mdn import MDN


def _loss(model, spectrum, design, epoch):
    return model.training_loss(spectrum, design)


def _builder(cls, **fixed):
    """build(num_res, spectrum_encoder="pooled") -> model predicting 2 * num_res positions."""

    def build(num_res: int, spectrum_encoder: str = POOLED):
        return cls(design_dim=design_dim(num_res), spectrum_encoder=spectrum_encoder, **fixed)

    return build


INVERSE_MODELS = {
    "1": {"name": "Mixture Density Network", "short": "MDN", "build": _builder(MDN, num_components=10),
          "loss_fn": _loss, "lr": 1e-3, "epochs": 100, "has_density": True},
    "2": {"name": "Conditional Normalizing Flow", "short": "Flow",
          "build": _builder(ConditionalFlow, num_layers=8, hidden=96),
          "loss_fn": _loss, "lr": 5e-4, "epochs": 100, "has_density": True},
    "3": {"name": "Conditional Diffusion", "short": "Diffusion",
          "build": _builder(ConditionalDiffusion, num_steps=100, hidden=128),
          "loss_fn": _loss, "lr": 1e-3, "epochs": 100, "has_density": False},
}
SHORT_TO_KEY = {spec["short"]: key for key, spec in INVERSE_MODELS.items()}
