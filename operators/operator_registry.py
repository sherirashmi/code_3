"""Central registry for ERP neural-operator experiment metadata.

Compatibility version:
- Works with the original operator files in this project.
- Operator modules only need to expose ``build_model`` and ``DEFAULT_MODEL_CONFIG``.
- Training defaults are centralized here, so operator files do NOT need
  ``DEFAULT_TRAINING_CONFIG`` and their local ``main()`` wrappers are not used.
"""

from __future__ import annotations

from utils.neural_operator_utils import make_operator_runner, random_search_operator

from operators.don import (
    build_model as build_don,
    DEFAULT_MODEL_CONFIG as DON_MODEL_CONFIG,
    SEARCH_SPACE as DON_SEARCH_SPACE,
)
from operators.dno import (
    build_model as build_dno,
    DEFAULT_MODEL_CONFIG as DNO_MODEL_CONFIG,
    SEARCH_SPACE as DNO_SEARCH_SPACE,
)
from operators.fno import (
    build_model as build_fno,
    DEFAULT_MODEL_CONFIG as FNO_MODEL_CONFIG,
    SEARCH_SPACE as FNO_SEARCH_SPACE,
)
from operators.dco import (
    build_model as build_dco,
    DEFAULT_MODEL_CONFIG as DCO_MODEL_CONFIG,
    SEARCH_SPACE as DCO_SEARCH_SPACE,
)
from operators.gno import (
    build_model as build_gno,
    DEFAULT_MODEL_CONFIG as GNO_MODEL_CONFIG,
    SEARCH_SPACE as GNO_SEARCH_SPACE,
)
from operators.set_transformer_operator import (
    build_model as build_sto,
    DEFAULT_MODEL_CONFIG as STO_MODEL_CONFIG,
    SEARCH_SPACE as STO_SEARCH_SPACE,
)
from operators.siren_operator import (
    build_model as build_siren,
    DEFAULT_MODEL_CONFIG as SIREN_MODEL_CONFIG,
    SEARCH_SPACE as SIREN_SEARCH_SPACE,
)
from operators.wno import (
    build_model as build_wno,
    DEFAULT_MODEL_CONFIG as WNO_MODEL_CONFIG,
    SEARCH_SPACE as WNO_SEARCH_SPACE,
)


def _make_spec(
    *,
    name: str,
    short: str,
    operator_name: str,
    build_model,
    model_config,
    search_space,
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
        "search_space": search_space,
        "epochs": int(epochs),
        "lr": float(lr),
    }


OPERATORS = {
    "1": _make_spec(
        name="DeepONet",
        short="DON",
        operator_name="DON",
        build_model=build_don,
        model_config=DON_MODEL_CONFIG,
        search_space=DON_SEARCH_SPACE,
        epochs=200,
        lr=5e-4,
    ),
    "2": _make_spec(
        name="Deep Neural Operator",
        short="DNO",
        operator_name="DNO",
        build_model=build_dno,
        model_config=DNO_MODEL_CONFIG,
        search_space=DNO_SEARCH_SPACE,
        epochs=200,
        lr=5e-4,
    ),
    "3": _make_spec(
        name="Fourier Neural Operator",
        short="FNO",
        operator_name="FNO",
        build_model=build_fno,
        model_config=FNO_MODEL_CONFIG,
        search_space=FNO_SEARCH_SPACE,
        epochs=200,
        lr=5e-4,
    ),
    "4": _make_spec(
        name="Deep Cat Operator",
        short="DCO",
        operator_name="DCO",
        build_model=build_dco,
        model_config=DCO_MODEL_CONFIG,
        search_space=DCO_SEARCH_SPACE,
        epochs=200,
        lr=5e-4,
    ),
    "5": _make_spec(
        name="Graph Neural Operator",
        short="GNO",
        operator_name="GNO",
        build_model=build_gno,
        model_config=GNO_MODEL_CONFIG,
        search_space=GNO_SEARCH_SPACE,
        epochs=200,
        lr=5e-4,
    ),
    "6": _make_spec(
        name="Set Transformer Operator",
        short="STO",
        operator_name="STO",
        build_model=build_sto,
        model_config=STO_MODEL_CONFIG,
        search_space=STO_SEARCH_SPACE,
        epochs=200,
        lr=5e-4,
    ),
    "7": _make_spec(
        name="SIREN Neural Operator",
        short="SIREN",
        operator_name="SIREN_NO",
        build_model=build_siren,
        model_config=SIREN_MODEL_CONFIG,
        search_space=SIREN_SEARCH_SPACE,
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
        search_space=WNO_SEARCH_SPACE,
        epochs=200,
        lr=5e-4,
    ),
}


def tune_operator(short: str, **search_kwargs) -> dict[str, object]:
    """Random-search one registered operator's own hyperparameters.

    ``short`` is the registry key used in the interactive menu (e.g. "DON",
    "FNO"). Every ``DEFAULT_MODEL_CONFIG`` entry that isn't part of the
    operator's ``SEARCH_SPACE`` (typically the parameter-matched
    width/hidden_dim) is held fixed, so tuning explores an architecture's
    own knobs without reopening the cross-architecture capacity match.
    Extra ``search_kwargs`` (e.g. ``num_trials``, ``search_epochs``) are
    forwarded to ``random_search_operator``.

    Example
    -------
        from operators.operator_registry import tune_operator
        result = tune_operator("FNO", num_trials=20, search_epochs=60)
        print(result["best_model_config"])
    """
    spec = next((s for s in OPERATORS.values() if s["short"] == short), None)
    if spec is None:
        valid = sorted(s["short"] for s in OPERATORS.values())
        raise ValueError(f"Unknown operator '{short}'. Choose one of: {valid}")

    fixed_config = {
        key: value
        for key, value in spec["model_config"].items()
        if key not in spec["search_space"]
    }
    return random_search_operator(
        operator_name=spec["short"],
        build_model=spec["build_model"],
        search_space=spec["search_space"],
        fixed_config=fixed_config,
        **search_kwargs,
    )


__all__ = ["OPERATORS", "tune_operator"]
