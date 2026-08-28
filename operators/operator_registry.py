"""Central registry for ERP neural-operator experiment metadata.

Compatibility version:
- Works with the original operator files in this project.
- Operator modules only need to expose ``build_model`` and ``DEFAULT_MODEL_CONFIG``.
- Training defaults are centralized here, so operator files do NOT need
  ``DEFAULT_TRAINING_CONFIG`` and their local ``main()`` wrappers are not used.
"""

from __future__ import annotations

from utils.neural_operator_utils import make_operator_runner

from operators.don import build_model as build_don, DEFAULT_MODEL_CONFIG as DON_MODEL_CONFIG
from operators.dno import build_model as build_dno, DEFAULT_MODEL_CONFIG as DNO_MODEL_CONFIG
from operators.fno import build_model as build_fno, DEFAULT_MODEL_CONFIG as FNO_MODEL_CONFIG
from operators.dco import build_model as build_dco, DEFAULT_MODEL_CONFIG as DCO_MODEL_CONFIG
from operators.gno import build_model as build_gno, DEFAULT_MODEL_CONFIG as GNO_MODEL_CONFIG
from operators.set_transformer_operator import (
    build_model as build_sto,
    DEFAULT_MODEL_CONFIG as STO_MODEL_CONFIG,
)
from operators.siren_operator import (
    build_model as build_siren,
    DEFAULT_MODEL_CONFIG as SIREN_MODEL_CONFIG,
)
from operators.wno import build_model as build_wno, DEFAULT_MODEL_CONFIG as WNO_MODEL_CONFIG


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
        epochs=500,
        lr=5e-3,
    ),
    "2": _make_spec(
        name="Deep Neural Operator",
        short="DNO",
        operator_name="DNO",
        build_model=build_dno,
        model_config=DNO_MODEL_CONFIG,
        epochs=500,
        lr=5e-3,
    ),
    "3": _make_spec(
        name="Fourier Neural Operator",
        short="FNO",
        operator_name="FNO",
        build_model=build_fno,
        model_config=FNO_MODEL_CONFIG,
        epochs=500,
        lr=5e-3,
    ),
    "4": _make_spec(
        name="Deep Cat Operator",
        short="DCO",
        operator_name="DCO",
        build_model=build_dco,
        model_config=DCO_MODEL_CONFIG,
        epochs=500,
        lr=5e-3,
    ),
    "5": _make_spec(
        name="Graph Neural Operator",
        short="GNO",
        operator_name="GNO",
        build_model=build_gno,
        model_config=GNO_MODEL_CONFIG,
        epochs=500,
        lr=5e-3,
    ),
    "6": _make_spec(
        name="Set Transformer Operator",
        short="STO",
        operator_name="STO",
        build_model=build_sto,
        model_config=STO_MODEL_CONFIG,
        epochs=500,
        lr=5e-3,
    ),
    "7": _make_spec(
        name="SIREN Neural Operator",
        short="SIREN",
        operator_name="SIREN_NO",
        build_model=build_siren,
        model_config=SIREN_MODEL_CONFIG,
        epochs=500,
        lr=2e-3,
    ),
    "8": _make_spec(
        name="Wavelet Neural Operator",
        short="WNO",
        operator_name="WNO",
        build_model=build_wno,
        model_config=WNO_MODEL_CONFIG,
        epochs=500,
        lr=5e-3,
    ),
}


__all__ = ["OPERATORS"]
