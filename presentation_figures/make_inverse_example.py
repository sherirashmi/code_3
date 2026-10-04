"""One real inverse-design example for the presentation figure.

Takes a held-out test spectrum of the 100k, 2-resonator grid dataset, samples 8 designs from the trained
Flow_b12 model, scores them (exact log p) and solves their ERP with the plate solver.
Writes presentation_figures/inverse_example.npz (nothing is written into the model/plot folders).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data
from erp_inverse.scripts.evaluate import _check_norm, load_inverse_model, score_samples
from utils.erp_dataset import DATASETS, configuration_to_resonators, denormalize_erp_array, select_dataset_modal_resolution
from utils.solver import compute_erp_spectrum

TAG = "100k_2res_grid_18modes"
KEY = "Flow_b12"
N_SAMPLES = 8
SEED = 727

select_dataset_modal_resolution(TAG)
files = list(DATASETS[TAG]["files"])
dataset, loaders = prepare_inverse_data(num_configurations=100000, batch_size=64, dataset_file=files, seed=SEED,
                                        design_param="bounded12")
norm = dataset.norm_params
freq_hz = np.asarray(dataset.frequency_values)
spectrum, design = next(iter(loaders["test"]))
target = denormalize_erp_array(spectrum[:1].numpy(), norm)[0]
true_cfg = denormalize_design(design[:1].reshape(1, -1).numpy(), int(norm["num_res"]), norm, consistent=False)[0]

model, model_norm = load_inverse_model(KEY, TAG)
_check_norm(KEY, model_norm, norm)
model.eval()
torch.manual_seed(SEED)
spec_t = torch.from_numpy(((target - norm["erp_mean"]) / norm["erp_std"]).astype(np.float32))[None, :]
with torch.no_grad():
    result = model.sample(spec_t, num_samples=N_SAMPLES)
flat = result[0] if isinstance(result, tuple) else result
physical, log_p = score_samples(model, spec_t, flat, norm)
physical, log_p = physical[0], log_p[0]  # (N, num_res, 5), (N,)
solved = np.stack([compute_erp_spectrum(configuration_to_resonators(np.asarray(c, dtype=np.float64)),
                                        frequencies=freq_hz) for c in physical])
mse = ((solved - target[None, :]) ** 2).mean(axis=1)
np.savez(ROOT / "presentation_figures" / "inverse_example.npz", freq=freq_hz, target=target, true_cfg=true_cfg,
         designs=physical, log_p=log_p, solved=solved, mse=mse)
print("true design [m,k,f_t,x,y]:\n", np.round(true_cfg, 3))
print("samples:\n", np.round(physical[..., [0, 2, 3, 4]].reshape(N_SAMPLES, -1), 3))
print("log p", np.round(log_p, 2), "\nsolver mse", np.round(mse, 2))
