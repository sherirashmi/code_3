"""Central registry for the 10 displacement forward operators, mirroring
forward_operators/operator_registry.py's shape (a key -> spec dict).
"""

from __future__ import annotations

from displacement_forward_operators.don import build_model as build_don, DEFAULT_MODEL_CONFIG as DON_MODEL_CONFIG
from displacement_forward_operators.dno import build_model as build_dno, DEFAULT_MODEL_CONFIG as DNO_MODEL_CONFIG
from displacement_forward_operators.fno import build_model as build_fno, DEFAULT_MODEL_CONFIG as FNO_MODEL_CONFIG
from displacement_forward_operators.dco import build_model as build_dco, DEFAULT_MODEL_CONFIG as DCO_MODEL_CONFIG
from displacement_forward_operators.gno import build_model as build_gno, DEFAULT_MODEL_CONFIG as GNO_MODEL_CONFIG
from displacement_forward_operators.set_transformer_operator import (
    build_model as build_sto,
    DEFAULT_MODEL_CONFIG as STO_MODEL_CONFIG,
)
from displacement_forward_operators.siren_operator import (
    build_model as build_siren,
    DEFAULT_MODEL_CONFIG as SIREN_MODEL_CONFIG,
)
from displacement_forward_operators.wno import build_model as build_wno, DEFAULT_MODEL_CONFIG as WNO_MODEL_CONFIG
from displacement_forward_operators.nn import build_model as build_nn, DEFAULT_MODEL_CONFIG as NN_MODEL_CONFIG
from displacement_forward_operators.lno import build_model as build_lno, DEFAULT_MODEL_CONFIG as LNO_MODEL_CONFIG

OPERATORS = {
    "1": {"name": "DeepONet", "short": "DON", "build_model": build_don, "model_config": DON_MODEL_CONFIG, "epochs": 100, "lr": 5e-4},
    "2": {"name": "Deep Neural Operator", "short": "DNO", "build_model": build_dno, "model_config": DNO_MODEL_CONFIG, "epochs": 100, "lr": 5e-4},
    "3": {"name": "Fourier Neural Operator", "short": "FNO", "build_model": build_fno, "model_config": FNO_MODEL_CONFIG, "epochs": 100, "lr": 5e-4},
    "4": {"name": "Deep Cat Operator", "short": "DCO", "build_model": build_dco, "model_config": DCO_MODEL_CONFIG, "epochs": 100, "lr": 5e-4},
    "5": {"name": "Graph Neural Operator", "short": "GNO", "build_model": build_gno, "model_config": GNO_MODEL_CONFIG, "epochs": 100, "lr": 5e-4},
    "6": {"name": "Set Transformer Operator", "short": "STO", "build_model": build_sto, "model_config": STO_MODEL_CONFIG, "epochs": 100, "lr": 5e-4},
    "7": {"name": "SIREN Neural Operator", "short": "SIREN", "build_model": build_siren, "model_config": SIREN_MODEL_CONFIG, "epochs": 100, "lr": 2e-4},
    "8": {"name": "Wavelet Neural Operator", "short": "WNO", "build_model": build_wno, "model_config": WNO_MODEL_CONFIG, "epochs": 100, "lr": 5e-4},
    "9": {"name": "Plain Neural Network", "short": "NN", "build_model": build_nn, "model_config": NN_MODEL_CONFIG, "epochs": 100, "lr": 5e-4},
    "10": {"name": "Laplace Neural Operator", "short": "LNO", "build_model": build_lno, "model_config": LNO_MODEL_CONFIG, "epochs": 100, "lr": 5e-4},
}
