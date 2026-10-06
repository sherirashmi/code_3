"""Central registry for ERP neural-operator experiment metadata.

Compatibility version:
- Works with the original operator files in this project.
- Operator modules only need to expose ``build_model`` and ``DEFAULT_MODEL_CONFIG``.
- Training defaults are centralized here, so operator files do NOT need
  ``DEFAULT_TRAINING_CONFIG`` and their local ``main()`` wrappers are not used.
"""

from __future__ import annotations

from erp_forward.scripts.neural_operator_utils import make_operator_runner

from erp_forward.scripts.don import build_model as build_don, DEFAULT_MODEL_CONFIG as DON_MODEL_CONFIG
from erp_forward.scripts.dno import build_model as build_dno, DEFAULT_MODEL_CONFIG as DNO_MODEL_CONFIG
from erp_forward.scripts.fno import build_model as build_fno, DEFAULT_MODEL_CONFIG as FNO_MODEL_CONFIG
from erp_forward.scripts.dco import build_model as build_dco, DEFAULT_MODEL_CONFIG as DCO_MODEL_CONFIG
from erp_forward.scripts.gno import build_model as build_gno, DEFAULT_MODEL_CONFIG as GNO_MODEL_CONFIG
from erp_forward.scripts.set_transformer_operator import (
    build_model as build_sto,
    DEFAULT_MODEL_CONFIG as STO_MODEL_CONFIG,
)
from erp_forward.scripts.siren_operator import (
    build_model as build_siren,
    DEFAULT_MODEL_CONFIG as SIREN_MODEL_CONFIG,
)
from erp_forward.scripts.wno import build_model as build_wno, DEFAULT_MODEL_CONFIG as WNO_MODEL_CONFIG
from erp_forward.scripts.nn import build_model as build_nn, DEFAULT_MODEL_CONFIG as NN_MODEL_CONFIG
from erp_forward.scripts.lno import build_model as build_lno, DEFAULT_MODEL_CONFIG as LNO_MODEL_CONFIG


# Architectures whose configuration encoder is ResonatorSetEncoder; GNO, STO
# and NN encode resonators differently and have no sorted-branch variant.
SORTED_BRANCH_OPERATORS = ("DON", "DNO", "FNO", "DCO", "SIREN", "WNO", "LNO")
SORTED_SUFFIX = "_sorted"

# Optional, opt-in forward variants (defaults reproduce every existing
# checkpoint). Each one adds a suffix to the model name, so variants are
# saved and plotted next to -- never over -- the standard models.
#   physical     coordinate_features="physical": modal sine features on
#                x/Lx, y/Ly and one detuning scale (f - f_t)/freq_std.
#                Every architecture except NN builds these features.
#   permutation  random resonator order every training batch (NN only; all
#                other architectures are permutation invariant by design).
#   padding_mode FFT frequency-axis padding (FNO only).
PHYSICAL_FEATURE_OPERATORS = ("DON", "DNO", "FNO", "DCO", "GNO", "STO", "SIREN", "WNO", "LNO")
PERMUTATION_AUGMENT_OPERATORS = ("NN",)
PADDING_MODE_OPERATORS = ("FNO",)
PHYSICAL_SUFFIX = "_phys"
PERMUTATION_SUFFIX = "_perm"
PADDING_SUFFIXES = {"replicate": "", "reflect": "_reflect", "zero": "_zpad"}


def forward_variant(spec, options=None) -> tuple[str, dict[str, object]]:
    """(model name, model_config overrides) of a forward-operator variant.

    ``options`` keys (all optional): ``sorted`` (bool), ``physical`` (bool),
    ``permutation`` (bool), ``padding_mode`` (str). Options an architecture
    does not support are ignored for it, e.g. ``{"sorted": True,
    "physical": True}`` gives ``DNO_sorted_phys`` but ``GNO_phys``.
    A plain bool is accepted as ``{"sorted": bool}`` (older call sites).
    """
    if isinstance(options, bool) or options is None:
        options = {"sorted": bool(options)}
    short = str(spec["short"])
    name, overrides = short, {}
    if options.get("sorted") and short in SORTED_BRANCH_OPERATORS:
        name += SORTED_SUFFIX
        overrides["use_sorted_branch"] = True
    if options.get("physical") and short in PHYSICAL_FEATURE_OPERATORS:
        name += PHYSICAL_SUFFIX
        overrides["coordinate_features"] = "physical"
    if options.get("permutation") and short in PERMUTATION_AUGMENT_OPERATORS:
        name += PERMUTATION_SUFFIX
        overrides["permutation_augment"] = True
    padding_mode = str(options.get("padding_mode") or "replicate")
    if padding_mode != "replicate" and short in PADDING_MODE_OPERATORS:
        name += PADDING_SUFFIXES[padding_mode]
        overrides["padding_mode"] = padding_mode
    return name, overrides


def _make_spec(
    *,
    name: str,
    short: str,
    operator_name: str,
    build_model,
    model_config,
    epochs: int,
    lr: float,
) -> dict[str, object]:
    """Create one operator registry entry and its common experiment runner."""
    runner = make_operator_runner(
        operator_name=operator_name,
        build_model=build_model,
        model_config=model_config,
        default_epochs=epochs,
        default_learning_rate=lr,
    )
    return {
        "name": name,
        "short": short,
        "runner": runner,
        "build_model": build_model,
        "model_config": model_config,
        "epochs": int(epochs),
        "lr": float(lr),
        # Accepts use_sorted_branch=True (pooled set encoder + f_t-sorted
        # resonator branch, the DCO_sorted design)?
        "supports_sorted_branch": short in SORTED_BRANCH_OPERATORS,
    }


OPERATORS = {
    "1": _make_spec(
        name="DeepONet",
        short="DON",
        operator_name="DON",
        build_model=build_don,
        model_config=DON_MODEL_CONFIG,
        epochs=200,
        lr=5e-4,
    ),
    "2": _make_spec(
        name="Deep Neural Operator",
        short="DNO",
        operator_name="DNO",
        build_model=build_dno,
        model_config=DNO_MODEL_CONFIG,
        epochs=200,
        lr=5e-4,
    ),
    "3": _make_spec(
        name="Fourier Neural Operator",
        short="FNO",
        operator_name="FNO",
        build_model=build_fno,
        model_config=FNO_MODEL_CONFIG,
        epochs=200,
        lr=5e-4,
    ),
    "4": _make_spec(
        name="Deep Cat Operator",
        short="DCO",
        operator_name="DCO",
        build_model=build_dco,
        model_config=DCO_MODEL_CONFIG,
        epochs=200,
        lr=5e-4,
    ),
    "5": _make_spec(
        name="Graph Neural Operator",
        short="GNO",
        operator_name="GNO",
        build_model=build_gno,
        model_config=GNO_MODEL_CONFIG,
        epochs=200,
        lr=5e-4,
    ),
    "6": _make_spec(
        name="Set Transformer Operator",
        short="STO",
        operator_name="STO",
        build_model=build_sto,
        model_config=STO_MODEL_CONFIG,
        epochs=200,
        lr=5e-4,
    ),
    "7": _make_spec(
        name="SIREN Neural Operator",
        short="SIREN",
        operator_name="SIREN_NO",
        build_model=build_siren,
        model_config=SIREN_MODEL_CONFIG,
        # Same epoch budget as every other operator so the comparison isn't
        # confounded by extra training time (was 250 vs. everyone else's 200).
        # lr stays lower than the shared 5e-4 default: sine layers are known
        # to need a smaller learning rate than smooth activations for stable
        # optimization (Sitzmann et al., SIREN) -- this is a documented,
        # architecture-intrinsic need, not an unfair training-budget edge.
        epochs=200,
        lr=2e-4,
    ),
    "8": _make_spec(
        name="Wavelet Neural Operator",
        short="WNO",
        operator_name="WNO",
        build_model=build_wno,
        model_config=WNO_MODEL_CONFIG,
        epochs=200,
        lr=5e-4,
    ),
    "9": _make_spec(
        name="Plain Neural Network",
        short="NN",
        operator_name="NN",
        build_model=build_nn,
        model_config=NN_MODEL_CONFIG,
        epochs=200,
        lr=5e-4,
    ),
    "10": _make_spec(
        name="Laplace Neural Operator",
        short="LNO",
        operator_name="LNO",
        build_model=build_lno,
        model_config=LNO_MODEL_CONFIG,
        epochs=200,
        lr=5e-4,
    ),
}


__all__ = [
    "OPERATORS",
    "SORTED_BRANCH_OPERATORS",
    "SORTED_SUFFIX",
    "PHYSICAL_FEATURE_OPERATORS",
    "PERMUTATION_AUGMENT_OPERATORS",
    "PADDING_MODE_OPERATORS",
    "PADDING_SUFFIXES",
    "forward_variant",
]
