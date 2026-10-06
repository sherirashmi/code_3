"""Does the inverse really use the target ERP? The inverse models are scored on N held-out test targets (200k_2res_18modes) against baselines that
IGNORE the target: (a) the mean ERP of the training set, (b) the bare plate (resonators too light to act), (c) designs drawn at random from the
training distribution (one design; best of 16). Every design is solved with the plate solver.
Skill = 1 - SSE(model) / SSE(baseline): above 0 the model is closer to the target than the baseline.
Saves inverse_baselines.csv / inverse_baselines.txt.

    python presentation_figures/prediction_cases/inverse_baselines.py [N]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403

import csv
import os
from concurrent.futures import ProcessPoolExecutor

from erp_forward.scripts.neural_operator_utils import _configuration_features, _split_ids
from erp_inverse.scripts.common import denormalize_design, prepare_inverse_data, sort_resonators_by_ft
from erp_inverse.scripts.evaluate import SPAWN_CONTEXT, solve_configs
from erp_invertible.scripts.registry import INVERTIBLE_OPERATORS
from erp_invertible.scripts.train import _design_dim_of
from erp_invertible_deeponet.scripts.train import load_variant
from utils.erp_dataset import DATASETS, configuration_to_resonators, normalize_erp_array, select_dataset_modal_resolution
from utils.paths import invertible_model_path
from utils.solver import compute_erp_spectrum
from utils.support import device

TAG = "200k_2res_18modes"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
S = 16
MODELS = [("iDON Q64", "idon", "Q64"), ("iDON Q64-ERP", "idon", "Q64-ERP"),
          ("iFNO", "ifno", "ifno_sortenc_phys_b12_bg_bin_pad_cyc_s2e"), ("iLNO", "ifno", "ilno_sortenc_phys_b12_bg_bin_cyc_s2e"),
          ("iSTO", "ifno", "isto_phys_b12_bg_bin_cyc_s2e"), ("iGNO", "ifno", "igno_phys_b12_bg_bin_cyc_s2e")]

if __name__ == "__main__":
    select_dataset_modal_resolution(TAG)
    files = list(DATASETS[TAG]["files"])
    dataset, _ = prepare_inverse_data(num_configurations=200000, batch_size=128, dataset_file=files, seed=727, design_param="bounded12")
    norm, num_res = dataset.norm_params, int(dataset.norm_params["num_res"])
    freq = np.asarray(dataset.frequency_values, dtype=np.float64)
    split = _split_ids(dataset)
    responses = np.asarray(dataset.responses, dtype=np.float64)[..., 0]
    target = responses[split["test"][:N]]
    mean_erp = responses[split["train"][:20000]].mean(0)
    bare = compute_erp_spectrum(configuration_to_resonators(build_configuration([(1e-4, 100.0, 0.7, 0.25), (1e-4, 100.0, 0.9, 0.25)])), frequencies=freq)
    train_feats = _configuration_features(dataset)[split["train"]].astype(np.float64)
    rng = np.random.default_rng(727)
    rand = sort_resonators_by_ft(train_feats[rng.integers(0, len(train_feats), size=(N, S))].reshape(-1, num_res, 5)).reshape(N, S, num_res, 5)
    spectra_n = torch.from_numpy(normalize_erp_array(target.astype(np.float32), norm))

    pool = ProcessPoolExecutor(max_workers=max(1, os.cpu_count() or 1), mp_context=SPAWN_CONTEXT)
    spec_by_short = {s["short"]: s for s in INVERTIBLE_OPERATORS.values()}
    results = {}
    solved_rand = solve_configs(pool, rand.reshape(-1, num_res, 5), freq).reshape(N, S, -1)
    results["random design (1)"] = (solved_rand[:, 0], None)
    results["random designs (best of 16)"] = (solved_rand[np.arange(N), ((solved_rand - target[:, None]) ** 2).mean(-1).argmin(1)], None)
    for label, kind, name in MODELS:
        if kind == "idon":
            model, mnorm = load_variant(name, TAG)
        else:
            ck = torch.load(invertible_model_path(name, TAG), map_location=device, weights_only=False)
            cfg = dict(ck["model_config"])
            model = spec_by_short[label]["build"](design_dim=_design_dim_of(cfg), **cfg)
            model.load_state_dict(ck["model_state_dict"])
            model.to(device).eval()
            mnorm = norm
        torch.manual_seed(727)
        spec = torch.from_numpy(normalize_erp_array(target.astype(np.float32), mnorm)).to(device)
        with torch.no_grad():
            if kind == "idon":
                flat = model.sample(spec, S).cpu().numpy()  # (N, S, D); sample 0 = point estimate
                point = flat[:, 0]
            else:
                point = model.infer_point_estimate(spec).cpu().numpy()
                flat = model.sample(spec, S).cpu().numpy()
        designs = sort_resonators_by_ft(denormalize_design(flat, num_res, mnorm).reshape(-1, num_res, 5)).reshape(N, S, num_res, 5)
        point_d = sort_resonators_by_ft(denormalize_design(point, num_res, mnorm))
        solved = solve_configs(pool, designs.reshape(-1, num_res, 5), freq).reshape(N, S, -1)
        solved_point = solved[:, 0] if kind == "idon" else solve_configs(pool, point_d, freq)
        best = solved[np.arange(N), ((solved - target[:, None]) ** 2).mean(-1).argmin(1)]
        results[f"{label} (point estimate)"] = (solved_point, point_d)
        results[f"{label} (best of 16)"] = (best, None)
        print(label, "done", flush=True)
    pool.shutdown()

    def sse(p):
        return ((p - target) ** 2).sum()

    base = {"mean training ERP": np.broadcast_to(mean_erp, target.shape), "bare plate": np.broadcast_to(bare, target.shape)}
    rows = []
    for name, p in {**base, **{k: v[0] for k, v in results.items()}}.items():
        rows.append(dict(method=name, rmse=float(np.sqrt(((p - target) ** 2).mean())), median_rmse=float(np.median(np.sqrt(((p - target) ** 2).mean(1)))),
                         skill_vs_mean=1 - sse(p) / sse(base["mean training ERP"]), skill_vs_bare=1 - sse(p) / sse(base["bare plate"])))
    lines = [f"Inverse models vs baselines that ignore the target ({N} held-out test targets of {TAG}; every design solved with the plate solver)",
             f"{'method':<34}{'RMSE (dB)':>10}{'median RMSE':>13}{'skill vs mean ERP':>19}{'skill vs bare plate':>21}"]
    lines += [f"{r['method']:<34}{r['rmse']:>10.2f}{r['median_rmse']:>13.2f}{r['skill_vs_mean']:>19.2f}{r['skill_vs_bare']:>21.2f}" for r in rows]
    (OUT / "inverse_baselines.txt").write_text("\n".join(lines) + "\n")
    with open(OUT / "inverse_baselines.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "rmse_db", "median_rmse_db", "skill_vs_mean_erp", "skill_vs_bare_plate"])
        w.writerows([[r["method"], round(r["rmse"], 3), round(r["median_rmse"], 3), round(r["skill_vs_mean"], 3), round(r["skill_vs_bare"], 3)] for r in rows])
    print("\n".join(lines))
