"""Central registry for the 7 trainable inverse models (6 probabilistic +
the deterministic, forward-surrogate-in-the-loop SurrogateInverse).

Mirrors ``erp_forward/scripts/operator_registry.py``'s shape (a key -> spec dict) so
the CLI can select "just MDN" or "all inverse models" the same way it
already selects individual forward operators or "all forward operators".
"""

from __future__ import annotations

from erp_inverse.scripts.basis_flow import BasisFlow
from erp_inverse.scripts.cvae import ConditionalVAE
from erp_inverse.scripts.diffusion import ConditionalDiffusion
from erp_inverse.scripts.flow import ConditionalFlow
from erp_inverse.scripts.mdn import MDN
from erp_inverse.scripts.padding_inn import PadINN

from erp_inverse.scripts.common import POOLED, POSITIONAL
from erp_inverse.scripts.design_space import BOUNDED12, FULL15, design_dim

# Default resonators per configuration. Every model is built for the number
# of resonators of its training dataset (``num_res`` argument of each
# builder); this is only the fallback for checkpoints that predate it.
NUM_RES = 3
DESIGN_DIM = NUM_RES * 5  # legacy full15 design size

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


def _build_surrogate(
    design_param: str = FULL15, spectrum_encoder: str = POOLED, dataset_tag: str | None = None, num_res: int = NUM_RES
):
    from erp_inverse.scripts.surrogate_inverse import (
        DEFAULT_SURROGATE_CHECKPOINTS,
        SurrogateInverse,
        surrogate_checkpoints_for,
    )

    checkpoints = surrogate_checkpoints_for(dataset_tag) if dataset_tag else DEFAULT_SURROGATE_CHECKPOINTS
    return SurrogateInverse(
        design_dim=design_dim(design_param, num_res), surrogate_checkpoints=checkpoints,
        spectrum_encoder=spectrum_encoder, num_res=num_res,
    )


def _builder(cls, *, uses_spectrum_encoder: bool = True, **fixed):
    """build(design_param="full15", spectrum_encoder="pooled", dataset_tag=None, num_res=3).

    ``num_res`` (resonators per configuration) sets the design size:
    5 * num_res (full15) or 4 * num_res (bounded12). Training passes the
    dataset's own value; loading passes the checkpoint's.
    """

    def build(
        design_param: str = FULL15, spectrum_encoder: str = POOLED, dataset_tag: str | None = None,
        num_res: int = NUM_RES,
    ):
        kwargs = dict(fixed, design_dim=design_dim(design_param, int(num_res)))
        if uses_spectrum_encoder:
            kwargs["spectrum_encoder"] = spectrum_encoder
        return cls(**kwargs)

    build.uses_spectrum_encoder = uses_spectrum_encoder
    return build


# --------------------------------------------------------------------------
# Model variants: spectrum encoder (pooled | positional) x design space
# (full15 | bounded12). The variant is part of the model's name, e.g.
# "Flow", "Flow_pos", "Flow_b12", "Flow_pos_b12", so checkpoints/plots of
# different variants never overwrite each other.
# --------------------------------------------------------------------------

_POS_SUFFIX = "_pos"
_B12_SUFFIX = "_b12"


def variant_name(short: str, spectrum_encoder: str = POOLED, design_param: str = FULL15) -> str:
    return short + (_POS_SUFFIX if spectrum_encoder == POSITIONAL else "") + (_B12_SUFFIX if design_param == BOUNDED12 else "")


def parse_variant(name: str) -> tuple[str, str, str]:
    """``"Flow_pos_b12"`` -> ``("Flow", "positional", "bounded12")``."""
    design_param = BOUNDED12 if name.endswith(_B12_SUFFIX) else FULL15
    if design_param == BOUNDED12:
        name = name[: -len(_B12_SUFFIX)]
    spectrum_encoder = POSITIONAL if name.endswith(_POS_SUFFIX) else POOLED
    if spectrum_encoder == POSITIONAL:
        name = name[: -len(_POS_SUFFIX)]
    return name, spectrum_encoder, design_param


INVERSE_MODELS = {
    "1": {
        "name": "Mixture Density Network",
        "short": "MDN",
        "build": _builder(MDN, num_components=10),
        "loss_fn": _mdn_loss,
        "lr": 1e-3,
        "epochs": 150,
    },
    "2": {
        "name": "Conditional VAE",
        "short": "cVAE",
        "build": _builder(ConditionalVAE, latent_dim=8),
        "loss_fn": _cvae_loss,
        "lr": 1e-3,
        "epochs": 150,
    },
    "3": {
        "name": "Conditional Normalizing Flow",
        "short": "Flow",
        "build": _builder(ConditionalFlow, num_layers=8, hidden=96),
        "loss_fn": _flow_loss,
        "lr": 5e-4,
        "epochs": 150,
    },
    "4": {
        "name": "Conditional Diffusion",
        "short": "Diffusion",
        "build": _builder(ConditionalDiffusion, num_steps=100, hidden=128),
        "loss_fn": _diffusion_loss,
        "lr": 1e-3,
        "epochs": 150,
    },
    "5": {
        "name": "Basis Flow (invertible, bidirectional)",
        "short": "BasisFlow",
        "build": _builder(BasisFlow, n_freq=301, num_layers=8, flow_hidden=96),
        "loss_fn": _basis_flow_loss,
        "lr": 5e-4,
        "epochs": 150,
    },
    "6": {
        "name": "Padding INN (Ardizzone et al., arXiv:1808.04730)",
        "short": "PadINN",
        "build": _builder(PadINN, uses_spectrum_encoder=False, n_freq=301, z_dim=16, num_layers=8, hidden=96),
        "loss_fn": _padding_inn_loss,
        "lr": 5e-4,
        "epochs": 150,
    },
    "7": {
        "name": "Surrogate-in-the-loop inverse (deterministic, frozen DCO+GNO)",
        "short": "Surrogate",
        "build": _build_surrogate,
        "loss_fn": _surrogate_loss,         # needs norm_params=..., see train_all
        "needs_norm_params": True,
        "lr": 5e-4,
        "epochs": 100,
    },
}

SHORT_TO_KEY = {spec["short"]: key for key, spec in INVERSE_MODELS.items()}

__all__ = ["INVERSE_MODELS", "SHORT_TO_KEY", "NUM_RES", "DESIGN_DIM", "variant_name", "parse_variant"]
