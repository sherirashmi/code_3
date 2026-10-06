"""True vs predicted ERP (forward direction) and true vs recovered design (inverse direction) of the invertible models on cases that were NOT
used for training (inference only). 2 resonators, 18 plate modes (200k_2res_18modes); ground truth = plate solver.

  iDON  Q64, Q64-ERP : erp_invertible_deeponet/models/200k_2res_18modes/idon_q64.pth, idon_q64-erp.pth
  iFNO family        : erp_invertible/models/200k_2res_18modes/<name>.pth (iFNO, iLNO, iSTO, iGNO)

Forward: design -> ERP (true design given). Inverse: target ERP -> 16 designs (sample 0 = point estimate); each is solved with the plate solver
and compared with the target ("best of 16" = oracle pick with the solver). The out-of-range case has no forward panel: its design cannot be
encoded by the models (bounded coordinates).

    python presentation_figures/prediction_cases/inverse_cases.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import utils.plot_style  # noqa: F401
from erp_forward.scripts.neural_operator_utils import _configuration_features, _split_ids
from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_invertible.scripts.registry import INVERTIBLE_OPERATORS
from erp_invertible.scripts.train import _design_dim_of
from erp_invertible_deeponet.scripts.train import load_variant, model_path
from utils.erp_dataset import (DATASETS, configuration_to_resonators, denormalize_erp_array, normalize_erp_array,
                               select_dataset_modal_resolution)
from utils.paths import invertible_model_path
from utils.physics import freqs
from utils.solver import compute_erp_spectrum
from utils.support import device
from erp_inverse.scripts.design_space import encode_bounded

TAG = "200k_2res_18modes"
TEST_INDEX = 3
NUM_SAMPLES = 16
# label, kind, checkpoint name, colour
MODELS = [
    ("iDON Q64", "idon", "Q64", "#6a3d9a"), ("iDON Q64-ERP", "idon", "Q64-ERP", "#b15928"),
    ("iFNO", "ifno", "ifno_sortenc_phys_b12_bg_bin_pad_cyc_s2e", "#d62728"), ("iLNO", "ifno", "ilno_sortenc_phys_b12_bg_bin_cyc_s2e", "#2ca02c"),
    ("iSTO", "ifno", "isto_phys_b12_bg_bin_cyc_s2e", "#1f77b4"), ("iGNO", "ifno", "igno_phys_b12_bg_bin_cyc_s2e", "#ff7f0e"),
]
NEW_CASES = [  # (title, [(m, f_t, x, y)], encodable)
    ("New configuration inside the training ranges (hand-picked, not in the dataset)", [(0.45, 55.0, 0.35, 0.15), (0.65, 110.0, 1.05, 0.38)], True),
    ("Outside the training ranges: heavy resonators, m = 2.0 and 1.5 kg (training: 0.1-1 kg); target ERP only",
     [(2.00, 60.0, 0.45, 0.25), (1.50, 120.0, 1.05, 0.15)], False),
    ("Different ERP: heavy absorbers on the first two plate peaks (46 and 55 Hz)", [(1.00, 46.0, 0.45, 0.25), (0.90, 55.0, 0.90, 0.28)], True),
    ("Different ERP: heavy absorbers on the two high plate peaks (98 and 145 Hz)", [(1.00, 98.0, 0.60, 0.20), (1.00, 145.0, 1.10, 0.30)], True),
    ("Different ERP: both resonators tuned to 100 Hz, heavy", [(1.00, 100.0, 0.55, 0.20), (1.00, 100.0, 0.90, 0.30)], True),
]

select_dataset_modal_resolution(TAG)
files = list(DATASETS[TAG]["files"])
dataset, loaders = prepare_inverse_data(num_configurations=int(DATASETS[TAG]["num_configurations"]), batch_size=128,
                                        dataset_file=files if len(files) > 1 else files[0], seed=727, design_param="bounded12")
norm = dataset.norm_params
num_res = int(norm["num_res"])
frequency_values = np.asarray(dataset.frequency_values, dtype=np.float64)
spectrum, design = next(iter(loaders["test"]))
i0 = TEST_INDEX - 1
test_design = denormalize_design(design[i0 : i0 + 1].reshape(1, -1).numpy(), num_res, norm, consistent=False)[0]
test_truth = denormalize_erp_array(spectrum[i0 : i0 + 1], norm)[0]
cases = [(f"Test-split configuration {TEST_INDEX} (held out: not used for training)", sort_resonators_by_ft(test_design), test_truth, True)]
for title, rows, ok in NEW_CASES:
    d = sort_resonators_by_ft(build_configuration(rows))
    cases.append((title, d, compute_erp_spectrum(configuration_to_resonators(d), frequencies=frequency_values), ok))
dist, ref = training_distance(dataset, np.stack([c[1] for c in cases]))

spec_by_short = {s["short"]: s for s in INVERTIBLE_OPERATORS.values()}
loaded, paths = {}, {}
for label, kind, name, _ in MODELS:
    if kind == "idon":
        model, mnorm = load_variant(name, TAG)
        paths[label] = model_path(name, TAG)
    else:
        paths[label] = invertible_model_path(name, TAG)
        ck = torch.load(paths[label], map_location=device, weights_only=False)
        cfg = dict(ck["model_config"])
        model = spec_by_short[label]["build"](design_dim=_design_dim_of(cfg), **cfg)
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()
        mnorm = norm
    loaded[label] = (model, mnorm, kind)


def solve(c):
    return compute_erp_spectrum(configuration_to_resonators(np.asarray(c, dtype=np.float64)), frequencies=frequency_values)


@torch.no_grad()
def forward_erp(label, design_phys):
    model, mnorm, kind = loaded[label]
    a = torch.from_numpy(encode_bounded(design_phys, mnorm).reshape(1, -1)).to(device)
    out = model(a) if kind == "idon" else model.predict_spectrum_from_design(a).squeeze(-1)
    return denormalize_erp_array(out.cpu().numpy(), mnorm)[0]


@torch.no_grad()
def inverse_designs(label, target):
    """(S, R, 5) designs sorted by f_t; row 0 = point estimate."""
    model, mnorm, kind = loaded[label]
    spec = torch.from_numpy(normalize_erp_array(target[None].astype(np.float32), mnorm)).to(device)
    torch.manual_seed(727)
    if kind == "idon":
        flat = model.sample(spec, NUM_SAMPLES)[0].cpu().numpy()
    else:
        point = model.infer_point_estimate(spec).cpu().numpy()
        flat = np.concatenate([point, model.sample(spec, NUM_SAMPLES)[0].cpu().numpy()], axis=0)  # row 0: point estimate, rows 1..16: samples
    return sort_resonators_by_ft(denormalize_design(flat, num_res, mnorm))


lines = ["Checkpoints used:"] + [f"  {label:<13} {paths[label]}" for label, *_ in MODELS]
lines += ["", f"Novelty: distance to the nearest TRAINING configuration (RMS of range-scaled m, f_t, x, y; test-configuration median {ref:.3f})", ""]
csv_rows = []
for n, (title, d, truth, encodable) in enumerate(cases, 1):
    notes = out_of_range(d)
    lines.append(f"Case {n}: {title}" + (f"  [outside: {'; '.join(notes)}]" if notes else ""))
    lines += [f"  true R{r + 1}: m={m:.2f} kg  f_t={f_t:.1f} Hz  (x, y)=({x:.2f}, {y:.2f}) m" for r, (m, k, f_t, x, y) in enumerate(d)]
    lines.append(f"  nearest training configuration: {dist[n - 1]:.3f}  ({dist[n - 1] / ref:.1f} x typical)")
    fwd, inv = {}, {}
    for label, *_ in MODELS:
        if encodable:
            fwd[label] = forward_erp(label, d)
        designs = inverse_designs(label, truth)
        solved = np.stack([solve(c) for c in designs])
        errs = np.sqrt(((solved - truth) ** 2).mean(axis=1))
        first = 0 if loaded[label][2] == "idon" else 1  # iDON: sample 0 is the point estimate and one of the 16; iFNO family: 16 extra samples
        best = first + int(errs[first : first + NUM_SAMPLES].argmin())
        inv[label] = dict(designs=designs, solved=solved, errs=errs, best=best)
        row = dict(case=n, model=label,
                   forward_rmse=rmse(fwd[label], truth) if encodable else None, forward_peak=peak_rmse(fwd[label], truth) if encodable else None,
                   point_rmse=float(errs[0]), best16_rmse=float(errs[best]))
        pe = []
        for pick, key in ((0, "point"), (best, "best16")):
            des = designs[pick]
            pe.append(100 * np.hypot(des[:, 3] - d[:, 3], des[:, 4] - d[:, 4]))
        row.update(point_pos_cm=pe[0].tolist(), best16_pos_cm=pe[1].tolist())
        csv_rows.append(row)
        lines.append(f"  {label:<13} forward RMSE {row['forward_rmse']:.2f} dB (peak {row['forward_peak']:.2f}) | " if encodable else f"  {label:<13} forward n/a | ")
        lines[-1] += (f"inverse: point-estimate ERP RMSE {errs[0]:.2f} dB, best of 16 {errs[best]:.2f} dB | "
                      f"best-of-16 design " + "; ".join(f"R{r + 1} m={inv[label]['designs'][best][r, 0]:.2f} f_t={inv[label]['designs'][best][r, 2]:.0f} "
                                                        f"({inv[label]['designs'][best][r, 3]:.2f}, {inv[label]['designs'][best][r, 4]:.2f}) [{pe[1][r]:.1f} cm]" for r in range(num_res)))
    lines.append("")
    colour = {label: c for label, _, _, c in MODELS}

    if encodable:  # figure 1: forward direction
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True)
        for ax, (label, *_ ) in zip(axes.flat, MODELS):
            ax.plot(frequency_values, truth, color="black", lw=2.2, label="True (solver)")
            ax.plot(frequency_values, fwd[label], color=colour[label], lw=1.7, label=f"{label} predicted")
            for ft in d[:, 2]:
                ax.axvline(ft, color="#c2412c", ls="-.", lw=0.8, zorder=0)
            ax.set_title(f"{label}  (RMSE {rmse(fwd[label], truth):.2f} dB)", fontsize=12)
            ax.grid(alpha=0.3)
            ax.set_xlim(freqs[0], freqs[-1])
        for ax in axes[1]:
            ax.set_xlabel("Frequency (Hz)")
        for ax in axes[:, 0]:
            ax.set_ylabel("ERP (dB)")
        axes[0, 0].legend(fontsize=9, loc="lower right")
        fig.suptitle(f"Forward direction, case {n}: {title}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / f"inverse_case_{n}_forward.png", dpi=170, facecolor="white")
        plt.close(fig)

    fig = plt.figure(figsize=(19, 9.4))  # figure 2: inverse direction
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], hspace=0.32, wspace=0.26, left=0.05, right=0.99, top=0.9, bottom=0.07)
    for i, (label, *_ ) in enumerate(MODELS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        r = inv[label]
        for s in range(1, len(r["solved"])):
            ax.plot(frequency_values, r["solved"][s], color="#b9bec6", lw=0.6, zorder=1)
        ax.plot([], [], color="#b9bec6", lw=0.8, label="other sampled designs")
        ax.plot(frequency_values, truth, color="black", lw=2.2, zorder=4, label="Target ERP")
        ax.plot(frequency_values, r["solved"][0], color=colour[label], lw=1.5, zorder=3, label=f"point estimate ({r['errs'][0]:.2f} dB)")
        ax.plot(frequency_values, r["solved"][r["best"]], color=colour[label], lw=1.9, ls="--", zorder=5, label=f"best of {NUM_SAMPLES} ({r['errs'][r['best']]:.2f} dB)")
        ax.set_title(label, fontsize=12)
        ax.grid(alpha=0.3)
        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_xlabel("Frequency (Hz)") if i >= 3 else None
        ax.set_ylabel("ERP of recovered designs, solver (dB)") if i % 3 == 0 else None
        ax.legend(fontsize=7.5, loc="lower right")
    axp = fig.add_subplot(gs[:, 3])
    draw_plate(axp, {"True": d}, marker_size=11)
    for label, *_ in MODELS:
        des = inv[label]["designs"][inv[label]["best"]]
        for r in range(num_res):
            axp.plot([d[r, 3], des[r, 3]], [d[r, 4], des[r, 4]], color=colour[label], lw=0.9, ls=":", zorder=3)
        axp.plot(des[:, 3], des[:, 4], "X", color=colour[label], ms=10, mec="black", mew=0.6, ls="", zorder=6)
    handles = [Line2D([], [], marker="o", color="#2a78d6", mec="black", ls="", ms=9, label="True resonators"),
               Line2D([], [], marker="*", color="#35d0ff", mec="black", ls="", ms=12, label="Force")]
    handles += [Line2D([], [], marker="X", color=colour[l], mec="black", mew=0.6, ls="", ms=9, label=f"{l} (best of {NUM_SAMPLES})") for l, *_ in MODELS]
    axp.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8.5)
    axp.set_title("Plate: true vs recovered resonators", fontsize=12)
    fig.suptitle(f"Inverse direction, case {n}: {title}" + (f"\n(outside the training set: {'; '.join(notes)})" if notes else ""), fontsize=13)
    fig.savefig(OUT / f"inverse_case_{n}_inverse.png", dpi=170, facecolor="white")
    plt.close(fig)

(OUT / "inverse_novelty.txt").write_text("\n".join(lines) + "\n")
with open(OUT / "inverse_cases.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["case", "model", "forward_rmse_db", "forward_peak_rmse_db", "inverse_point_rmse_db", "inverse_best16_rmse_db", "point_position_error_cm", "best16_position_error_cm"])
    for r in csv_rows:
        w.writerow([r["case"], r["model"], r["forward_rmse"] and round(r["forward_rmse"], 3), r["forward_peak"] and round(r["forward_peak"], 3),
                    round(r["point_rmse"], 3), round(r["best16_rmse"], 3), [round(v, 1) for v in r["point_pos_cm"]], [round(v, 1) for v in r["best16_pos_cm"]]])
print("\n".join(lines))
