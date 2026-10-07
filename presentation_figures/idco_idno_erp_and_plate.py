"""iDCO and iDNO (iFNO family, 200k_2res_18modes, final checkpoints): for the first five held-out test targets (the five examples of the stored
inverse_validation_reconstructions.png) the ERP of the sampled designs (plate solver) next to the plate with the true and the predicted resonators.

For every target 8 designs are sampled (seed 727); the "selected" design is the one whose own forward prediction is closest to the target ERP
(model's own forward direction, as in erp_invertible/scripts/train.py evaluate_inverse). Inference only. Saves
erp_invertible/plots/models/200k_2res_18modes/<model>/inverse_erp_and_plate.png (a new random draw: the selected designs can differ from the
ones in the stored inverse_validation_reconstructions.png).

    python presentation_figures/idco_idno_erp_and_plate.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import utils.plot_style  # noqa: F401
from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_invertible.scripts.train import RECOMMENDED_OPTIONS, _load_checkpoint, variant
from utils.erp_dataset import (DATASETS, configuration_to_resonators, denormalize_erp_array, normalize_configuration_array,
                               select_dataset_modal_resolution)
from utils.paths import invertible_plot_dir
from utils.physics import Lx, Ly, edge_margin, xf, yf
from utils.solver import compute_erp_spectrum
from utils.support import device

TAG = "200k_2res_18modes"
MODELS = [("2", "iDCO"), ("4", "iDNO")]
N_EXAMPLES, N_SAMPLES, SEED = 5, 8, 727
RES_COLORS = ("#2a78d6", "#eb6834")

select_dataset_modal_resolution(TAG)
files = list(DATASETS[TAG]["files"])
dataset, loaders = prepare_inverse_data(num_configurations=int(DATASETS[TAG]["num_configurations"]), batch_size=128,
                                        dataset_file=files if len(files) > 1 else files[0], seed=727, design_param="bounded12")
norm = dataset.norm_params
num_res = int(norm["num_res"])
freq = np.asarray(dataset.frequency_values, dtype=np.float64)
spectrum, design = next(iter(loaders["test"]))
spectrum, design = spectrum[:N_EXAMPLES], design[:N_EXAMPLES]
target = denormalize_erp_array(spectrum, norm)
true_design = sort_resonators_by_ft(denormalize_design(design.reshape(N_EXAMPLES, -1).numpy(), num_res, norm, consistent=False))


def solve(c):
    return compute_erp_spectrum(configuration_to_resonators(np.asarray(c, dtype=np.float64)), frequencies=freq)


for key, label in MODELS:
    model, ck = _load_checkpoint(key, TAG, False, RECOMMENDED_OPTIONS)
    model.set_design_normalization(norm)
    model.eval()
    name = variant(key, False, RECOMMENDED_OPTIONS)[0]
    torch.manual_seed(SEED)
    with torch.no_grad():
        samples = model.sample(spectrum.to(device), num_samples=N_SAMPLES).cpu()
    physical = denormalize_design(samples.numpy(), num_res, norm)  # (n, S, R, 5)
    with torch.no_grad():
        renorm = torch.from_numpy(normalize_configuration_array(physical, norm)).to(device)
        own = model.predict_spectrum(renorm.reshape(N_EXAMPLES * N_SAMPLES, num_res, 5)).squeeze(-1).reshape(N_EXAMPLES, N_SAMPLES, -1).cpu()
    selected = ((own - spectrum[:, None, :]) ** 2).mean(-1).argmin(1).numpy()

    fig = plt.figure(figsize=(19, 4.1 * N_EXAMPLES))
    gs = fig.add_gridspec(N_EXAMPLES, 2, width_ratios=[2.6, 1.5], hspace=0.62, wspace=0.12, left=0.055, right=0.985, top=0.93, bottom=0.05)
    for r in range(N_EXAMPLES):
        solved = np.stack([solve(c) for c in physical[r]])
        mse = ((solved - target[r]) ** 2).mean(1)
        sel = int(selected[r])
        axe = fig.add_subplot(gs[r, 0])
        for s in range(N_SAMPLES):
            if s != sel:
                axe.plot(freq, solved[s], color="#4C72B0", alpha=0.3, lw=1.0, label="Other samples" if s == (1 if sel == 0 else 0) else None)
        axe.plot(freq, solved[sel], color="#C44E52", lw=2.0, label="Selected sample (solver)")
        axe.plot(freq, target[r], color="black", lw=2.0, label="Target ERP")
        for ft in true_design[r][:, 2]:
            axe.axvline(ft, color="#c2412c", ls="-.", lw=0.7, zorder=0)
        axe.set_title(f"Example {r + 1}: MSE $= {mse[sel]:.2f}$ dB$^2$ (RMSE {np.sqrt(mse[sel]):.2f} dB)", fontsize=11)
        axe.set_xlim(freq[0], freq[-1])
        axe.grid(alpha=0.3)
        axe.set_ylabel("ERP (dB re 1 pW)")
        if r == N_EXAMPLES - 1:
            axe.set_xlabel("Frequency $f$ (Hz)")
        if r == 0:
            axe.legend(fontsize=9, loc="lower right")

        axp = fig.add_subplot(gs[r, 1])
        axp.add_patch(Rectangle((0, 0), Lx, Ly, facecolor="#f4f4f1", ec="black", lw=1.2, zorder=0))
        axp.add_patch(Rectangle((edge_margin, edge_margin), Lx - 2 * edge_margin, Ly - 2 * edge_margin, fill=False, ec="#9a9a94", ls="--", lw=0.8))
        axp.plot([xf], [yf], marker="*", color="#35d0ff", ms=15, mec="black", mew=0.8, ls="", zorder=6)
        t = true_design[r]
        pred = sort_resonators_by_ft(physical[r])[sel] if False else sort_resonators_by_ft(physical[r, sel][None])[0]
        others = sort_resonators_by_ft(physical[r])
        for s in range(N_SAMPLES):
            if s != sel:
                for i in range(num_res):
                    axp.plot(others[s, i, 3], others[s, i, 4], "o", color=RES_COLORS[i], ms=4.5, alpha=0.3, mec="none", zorder=2)
        lines = []
        for i in range(num_res):
            axp.plot([t[i, 3], pred[i, 3]], [t[i, 4], pred[i, 4]], color=RES_COLORS[i], lw=1.0, ls=":", zorder=3)
            axp.plot(t[i, 3], t[i, 4], "o", color=RES_COLORS[i], ms=13, mec="black", mew=1.0, zorder=5)
            axp.plot(pred[i, 3], pred[i, 4], "X", color=RES_COLORS[i], ms=12, mec="black", mew=0.8, zorder=6)
            err = 100 * np.hypot(t[i, 3] - pred[i, 3], t[i, 4] - pred[i, 4])
            lines.append(f"R{i + 1}: true m={t[i, 0]:.2f} kg, $f_t$={t[i, 2]:.0f} Hz | pred m={pred[i, 0]:.2f}, $f_t$={pred[i, 2]:.0f} | {err:.0f} cm")
        axp.set_xlim(-0.04, Lx + 0.04)
        axp.set_ylim(-0.04, Ly + 0.04)
        axp.set_aspect("equal")
        axp.set_xlabel("Position $x$ (m)")
        axp.set_ylabel("Position $y$ (m)")
        axp.set_title("Plate: true (circles) vs predicted (crosses)\n" + "\n".join(lines), fontsize=9.5, linespacing=1.35)
    handles = [Line2D([], [], marker="*", color="#35d0ff", mec="black", ls="", ms=12, label="Excitation force"),
               Line2D([], [], marker="o", color="#777777", mec="black", ls="", ms=10, label="True resonator"),
               Line2D([], [], marker="X", color="#777777", mec="black", ls="", ms=10, label="Selected prediction"),
               Line2D([], [], marker="o", color="#777777", alpha=0.4, ls="", ms=6, label="Other samples")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.78, 0.0))
    fig.suptitle(f"{label}: inverse design of held-out test targets, ERP of the sampled designs and resonator positions (colours: R1, R2 = lower, higher $f_t$)", fontsize=14)
    out = invertible_plot_dir(TAG, name) / "inverse_erp_and_plate.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print("saved", out)
