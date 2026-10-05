"""Forward ERP prediction of the invertible operators (erp_invertible/models/200k_2res_18modes) on one test configuration, all in one plot.

Each checkpoint is loaded as in erp_invertible/scripts/train.py (_load_checkpoint / evaluate_one); the configuration is the TEST_INDEX-th
test configuration of the 200k_2res_18modes split (index 3, as in the iDON inverse example). Ground truth: the test spectrum (solver).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.plot_style  # noqa: F401
from erp_inverse.scripts.common import prepare_inverse_data
from erp_invertible.scripts.registry import INVERTIBLE_OPERATORS
from erp_invertible.scripts.train import _design_dim_of
from utils.erp_dataset import DATASETS, denormalize_erp_array, select_dataset_modal_resolution
from utils.paths import invertible_model_path
from utils.support import device

TAG = "200k_2res_18modes"
TEST_INDEX = 3  # 1-based test configuration
MODELS = [  # label, short name in the registry, checkpoint name, colour
    ("iFNO", "iFNO", "ifno_sortenc_phys_b12_bg_bin_pad_cyc_s2e", "#d62728"),
    ("iLNO", "iLNO", "ilno_sortenc_phys_b12_bg_bin_cyc_s2e", "#2ca02c"),
    ("iSTO", "iSTO", "isto_phys_b12_bg_bin_cyc_s2e", "#1f77b4"),
    ("iGNO", "iGNO", "igno_phys_b12_bg_bin_cyc_s2e", "#ff7f0e"),
]
spec_by_short = {s["short"]: s for s in INVERTIBLE_OPERATORS.values()}

select_dataset_modal_resolution(TAG)
files = list(DATASETS[TAG]["files"])
dataset, loaders = prepare_inverse_data(num_configurations=int(DATASETS[TAG]["num_configurations"]), batch_size=128,
                                        dataset_file=files if len(files) > 1 else files[0], seed=727, design_param="bounded12")
spectrum, design = next(iter(loaders["test"]))
idx = TEST_INDEX - 1
truth = denormalize_erp_array(spectrum[idx : idx + 1], dataset.norm_params)[0]
freq = np.asarray(dataset.frequency_values)

curves = {}
for label, short, name, colour in MODELS:
    ck = torch.load(invertible_model_path(name, TAG), map_location=device, weights_only=False)
    cfg = dict(ck["model_config"])
    spec = spec_by_short[short]
    model = spec["build"](design_dim=_design_dim_of(cfg), **cfg)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    with torch.no_grad():
        pred = model.predict_spectrum_from_design(design[idx : idx + 1].to(device).reshape(1, -1)).squeeze(-1).cpu()
    curves[label] = denormalize_erp_array(pred, dataset.norm_params)[0]
    print(f"{label}: MSE {((curves[label] - truth) ** 2).mean():.3f} dB^2, MAE {np.abs(curves[label] - truth).mean():.3f} dB")

fig, ax = plt.subplots(figsize=(10.5, 5.2))
ax.plot(freq, truth, color="black", lw=2.8, label="Ground truth (solver)")
for (label, _, _, colour) in MODELS:
    ax.plot(freq, curves[label], color=colour, lw=1.6, ls="--", label=label)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("ERP (dB)")
ax.grid(True, color="#d9d9d4", lw=0.5)
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
fig.tight_layout()
for ext, kw in (("png", dict(dpi=220)), ("pdf", {}), ("svg", {})):
    fig.savefig(ROOT / "presentation_figures" / f"invertible_models_one_plot.{ext}", facecolor="white", **kw)
print("saved")
