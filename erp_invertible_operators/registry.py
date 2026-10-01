"""Central registry for the invertible forward+inverse operator family.

Mirrors erp_inverse_operators/registry.py's shape (a key -> spec dict) so
train.py can select "just iDCO" or "all three" the same way the rest of
this project's CLIs already select individual models or "all models".
"""

from __future__ import annotations

from erp_invertible_operators.idco import DEFAULT_MODEL_CONFIG as IDCO_CONFIG, IDCO, build_model as build_idco
from erp_invertible_operators.ifno import DEFAULT_MODEL_CONFIG as IFNO_CONFIG, IFNO, build_model as build_ifno
from erp_invertible_operators.igno import DEFAULT_MODEL_CONFIG as IGNO_CONFIG, IGNO, build_model as build_igno

INVERTIBLE_OPERATORS = {
    "1": {
        "name": "Invertible Fourier Neural Operator",
        "short": "iFNO",
        "supports_sorted_branch": True,
        "build": build_ifno,
        "model_config": IFNO_CONFIG,
        "model_cls": IFNO,
    },
    "2": {
        "name": "Invertible Deep Cat Operator",
        "short": "iDCO",
        "supports_sorted_branch": True,
        "build": build_idco,
        "model_config": IDCO_CONFIG,
        "model_cls": IDCO,
    },
    "3": {
        "name": "Invertible Graph Neural Operator",
        "short": "iGNO",
        "supports_sorted_branch": False,
        "build": build_igno,
        "model_config": IGNO_CONFIG,
        "model_cls": IGNO,
    },
}

__all__ = ["INVERTIBLE_OPERATORS"]
