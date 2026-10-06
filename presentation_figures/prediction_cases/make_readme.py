"""Rebuild README.md of this folder from the CSV / txt results written by forward_cases.py, inverse_cases.py and inverse_baselines.py."""
import csv
import re
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent


def titles(txt):
    return {int(m.group(1)): m.group(2).split("  [")[0] for m in re.finditer(r"^Case (\d+): (.*)$", (D / txt).read_text(), re.M)}


ft, it = titles("forward_novelty.txt"), titles("inverse_novelty.txt")
fw = defaultdict(dict)
for r in csv.DictReader(open(D / "forward_cases.csv")):
    fw[int(r["case"])][r["model"]] = float(r["rmse_db"])
models = list(next(iter(fw.values())))
fwd = ["| Case | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
fwd += [f"| {c}. {ft[c]} | " + " | ".join(f"{fw[c][m]:.2f}" for m in models) + " |" for c in sorted(fw)]
iv = defaultdict(dict)
for r in csv.DictReader(open(D / "inverse_cases.csv")):
    iv[int(r["case"])][r["model"]] = r
imodels = list(next(iter(iv.values())))
inv = ["| Case | Model | Forward RMSE (dB) | Inverse, point estimate (dB) | Inverse, best of 16 (dB) |", "|---|---|---|---|---|"]
for c in sorted(iv):
    for m in imodels:
        r = iv[c][m]
        f = f"{float(r['forward_rmse_db']):.2f}" if r["forward_rmse_db"] else "n/a"
        inv.append(f"| {c}. {it[c]} | {m} | {f} | {float(r['inverse_point_rmse_db']):.2f} | {float(r['inverse_best16_rmse_db']):.2f} |")
base = ["| Method | RMSE (dB) | Median RMSE (dB) | Skill vs mean ERP | Skill vs bare plate |", "|---|---|---|---|---|"]
for r in csv.DictReader(open(D / "inverse_baselines.csv")):
    base.append(f"| {r['method']} | {float(r['rmse_db']):.2f} | {float(r['median_rmse_db']):.2f} | {float(r['skill_vs_mean_erp']):.2f} | {float(r['skill_vs_bare_plate']):.2f} |")
(D / "README.md").write_text(f"""# True vs predicted: the final trained models on unseen cases

Inference only (nothing was retrained). Ground truth is always the plate solver. Scripts: `forward_cases.py`, `inverse_cases.py`, `inverse_baselines.py`
(shared helpers in `common.py`; this file is rebuilt by `make_readme.py`). Exact designs, per-model numbers and the distance of every case to the training set:
`forward_novelty.txt`, `inverse_novelty.txt`, `forward_cases.csv`, `inverse_cases.csv`.

## Checkpoints used

Forward operators, 3 resonators, 100k dataset (`erp_forward/models/100k/`): DON `don_sorted_phys.pth`, DNO `dno_sorted_phys.pth`, DCO `dco_sorted_phys.pth`,
FNO `fno_sorted_phys.pth`, WNO `wno_sorted_phys.pth`, LNO `lno_sorted_phys.pth`, GNO `gno_phys.pth`, STO `sto_phys.pth`, SIREN `siren_sorted_phys.pth`, NN `nn_perm.pth`.

Invertible models, 2 resonators, 200k dataset: iDON Q64 `erp_invertible_deeponet/models/200k_2res_18modes/idon_q64.pth`, iDON Q64-ERP `.../idon_q64-erp.pth`;
iFNO `erp_invertible/models/200k_2res_18modes/ifno_sortenc_phys_b12_bg_bin_pad_cyc_s2e.pth`, iLNO `ilno_sortenc_phys_b12_bg_bin_cyc_s2e.pth`,
iSTO `isto_phys_b12_bg_bin_cyc_s2e.pth`, iGNO `igno_phys_b12_bg_bin_cyc_s2e.pth`.

## Forward ERP prediction, RMSE vs solver (dB), figures `forward_case_<n>.png`

{chr(10).join(fwd)}

## Invertible models, figures `inverse_case_<n>_forward.png` (design -> ERP) and `inverse_case_<n>_inverse.png` (ERP -> design, checked with the solver)

{chr(10).join(inv)}

Forward: the true design is the input. Inverse: the target ERP is the input; each recovered design is solved with the plate solver and compared with the target.
"Best of 16" picks, with the solver and the target, the best of 16 sampled designs (an oracle choice, not available for a real target). The out-of-range case has no
forward panel because the models encode the design in bounded coordinates (m <= 1 kg), so a 2 kg resonator cannot be given to them.

## Does the inverse use the target? Baselines that ignore it (`inverse_baselines.py`, 200 held-out test targets)

{chr(10).join(base)}

Skill = 1 - SSE(method) / SSE(baseline): above 0 the method is closer to the target ERP than the baseline. The mean training ERP and the bare plate (resonators too
light to act) do not look at the target. Random designs are drawn from the training set and solved.
""")
