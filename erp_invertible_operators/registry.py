"""Central registry for the invertible forward+inverse operator family
(iFNO, iDCO, iGNO, iDNO, iWNO, iLNO, iSIREN, iSTO).

Mirrors erp_inverse_operators/registry.py's shape (a key -> spec dict) so
train.py can select "just iDCO" or "all three" the same way the rest of
this project's CLIs already select individual models or "all models".
"""

from __future__ import annotations

from erp_invertible_operators.idco import DEFAULT_MODEL_CONFIG as IDCO_CONFIG, IDCO, build_model as build_idco
from erp_invertible_operators.ifno import DEFAULT_MODEL_CONFIG as IFNO_CONFIG, IFNO, build_model as build_ifno
from erp_invertible_operators.igno import DEFAULT_MODEL_CONFIG as IGNO_CONFIG, IGNO, build_model as build_igno
from erp_invertible_operators.idno import DEFAULT_MODEL_CONFIG as IDNO_CONFIG, IDNO, build_model as build_idno
from erp_invertible_operators.ilno import DEFAULT_MODEL_CONFIG as ILNO_CONFIG, ILNO, build_model as build_ilno
from erp_invertible_operators.isiren import DEFAULT_MODEL_CONFIG as ISIREN_CONFIG, ISIREN, build_model as build_isiren
from erp_invertible_operators.isto import DEFAULT_MODEL_CONFIG as ISTO_CONFIG, ISTO, build_model as build_isto
from erp_invertible_operators.iwno import DEFAULT_MODEL_CONFIG as IWNO_CONFIG, IWNO, build_model as build_iwno

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
    "4": {
        "name": "Invertible Deep Neural Operator",
        "short": "iDNO",
        "supports_sorted_branch": True,
        "build": build_idno,
        "model_config": IDNO_CONFIG,
        "model_cls": IDNO,
    },
    "5": {
        "name": "Invertible Wavelet Neural Operator",
        "short": "iWNO",
        "supports_sorted_branch": True,
        "build": build_iwno,
        "model_config": IWNO_CONFIG,
        "model_cls": IWNO,
    },
    "6": {
        "name": "Invertible Laplace Neural Operator",
        "short": "iLNO",
        "supports_sorted_branch": True,
        "build": build_ilno,
        "model_config": ILNO_CONFIG,
        "model_cls": ILNO,
    },
    "7": {
        "name": "Invertible SIREN Neural Operator",
        "short": "iSIREN",
        "supports_sorted_branch": True,
        "build": build_isiren,
        "model_config": ISIREN_CONFIG,
        "model_cls": ISIREN,
    },
    "8": {
        "name": "Invertible Set Transformer Operator",
        "short": "iSTO",
        "supports_sorted_branch": False,
        "build": build_isto,
        "model_config": ISTO_CONFIG,
        "model_cls": ISTO,
    },
}

__all__ = ["INVERTIBLE_OPERATORS"]
