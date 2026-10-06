# True vs predicted: the final trained models on unseen cases

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

| Case | DON | DNO | DCO | FNO | WNO | LNO | GNO | STO | SIREN | NN |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Test-split configuration 3 (held out: not used for training) | 5.65 | 3.48 | 3.59 | 4.49 | 4.26 | 4.57 | 3.86 | 3.64 | 3.64 | 7.99 |
| 2. New configuration inside the training ranges (hand-picked, not in the dataset) | 7.31 | 1.74 | 1.97 | 2.80 | 3.65 | 2.14 | 3.65 | 3.50 | 2.33 | 7.72 |
| 3. All three resonators tuned to the same frequency (72 Hz), placed along one line | 8.23 | 3.72 | 2.55 | 4.70 | 4.94 | 5.26 | 5.84 | 7.84 | 3.22 | 9.63 |
| 4. Outside the training ranges: heavy resonators, m = 2.0, 1.5 and 2.5 kg (training: 0.1-1 kg) | 12.21 | 10.91 | 8.60 | 12.99 | 10.64 | 8.01 | 14.32 | 14.80 | 10.52 | 11.40 |
| 5. Different ERP: heavy absorbers on three plate peaks (1 kg at 46, 98 and 145 Hz) | 9.98 | 4.91 | 4.57 | 6.13 | 6.61 | 6.06 | 6.44 | 6.81 | 5.17 | 9.59 |
| 6. Different ERP: very light resonators (0.1 kg), almost the bare plate | 5.30 | 1.44 | 1.13 | 2.33 | 2.79 | 1.31 | 1.83 | 2.46 | 1.58 | 6.67 |

## Invertible models, figures `inverse_case_<n>_forward.png` (design -> ERP) and `inverse_case_<n>_inverse.png` (ERP -> design, checked with the solver)

| Case | Model | Forward RMSE (dB) | Inverse, point estimate (dB) | Inverse, best of 16 (dB) |
|---|---|---|---|---|
| 1. Test-split configuration 3 (held out: not used for training) | iDON Q64 | 2.02 | 6.41 | 6.41 |
| 1. Test-split configuration 3 (held out: not used for training) | iDON Q64-ERP | 3.56 | 5.05 | 5.04 |
| 1. Test-split configuration 3 (held out: not used for training) | iFNO | 1.52 | 8.63 | 7.57 |
| 1. Test-split configuration 3 (held out: not used for training) | iLNO | 1.43 | 10.71 | 9.19 |
| 1. Test-split configuration 3 (held out: not used for training) | iSTO | 1.53 | 9.50 | 9.52 |
| 1. Test-split configuration 3 (held out: not used for training) | iGNO | 2.79 | 11.27 | 7.38 |
| 2. New configuration inside the training ranges (hand-picked, not in the dataset) | iDON Q64 | 2.85 | 8.04 | 7.27 |
| 2. New configuration inside the training ranges (hand-picked, not in the dataset) | iDON Q64-ERP | 3.80 | 3.21 | 3.21 |
| 2. New configuration inside the training ranges (hand-picked, not in the dataset) | iFNO | 2.94 | 6.81 | 4.71 |
| 2. New configuration inside the training ranges (hand-picked, not in the dataset) | iLNO | 2.25 | 8.92 | 5.65 |
| 2. New configuration inside the training ranges (hand-picked, not in the dataset) | iSTO | 2.63 | 8.64 | 7.56 |
| 2. New configuration inside the training ranges (hand-picked, not in the dataset) | iGNO | 3.43 | 7.32 | 5.70 |
| 3. Outside the training ranges: heavy resonators, m = 2.0 and 1.5 kg (training: 0.1-1 kg); target ERP only | iDON Q64 | n/a | 10.35 | 9.07 |
| 3. Outside the training ranges: heavy resonators, m = 2.0 and 1.5 kg (training: 0.1-1 kg); target ERP only | iDON Q64-ERP | n/a | 13.01 | 9.92 |
| 3. Outside the training ranges: heavy resonators, m = 2.0 and 1.5 kg (training: 0.1-1 kg); target ERP only | iFNO | n/a | 10.78 | 9.40 |
| 3. Outside the training ranges: heavy resonators, m = 2.0 and 1.5 kg (training: 0.1-1 kg); target ERP only | iLNO | n/a | 13.57 | 8.96 |
| 3. Outside the training ranges: heavy resonators, m = 2.0 and 1.5 kg (training: 0.1-1 kg); target ERP only | iSTO | n/a | 10.60 | 9.45 |
| 3. Outside the training ranges: heavy resonators, m = 2.0 and 1.5 kg (training: 0.1-1 kg); target ERP only | iGNO | n/a | 9.51 | 8.89 |
| 4. Different ERP: heavy absorbers on the first two plate peaks (46 and 55 Hz) | iDON Q64 | 5.35 | 10.05 | 9.31 |
| 4. Different ERP: heavy absorbers on the first two plate peaks (46 and 55 Hz) | iDON Q64-ERP | 9.31 | 11.70 | 9.21 |
| 4. Different ERP: heavy absorbers on the first two plate peaks (46 and 55 Hz) | iFNO | 4.01 | 9.97 | 8.14 |
| 4. Different ERP: heavy absorbers on the first two plate peaks (46 and 55 Hz) | iLNO | 4.76 | 10.58 | 10.27 |
| 4. Different ERP: heavy absorbers on the first two plate peaks (46 and 55 Hz) | iSTO | 4.56 | 7.95 | 8.93 |
| 4. Different ERP: heavy absorbers on the first two plate peaks (46 and 55 Hz) | iGNO | 6.57 | 9.36 | 9.54 |
| 5. Different ERP: heavy absorbers on the two high plate peaks (98 and 145 Hz) | iDON Q64 | 5.93 | 13.60 | 10.38 |
| 5. Different ERP: heavy absorbers on the two high plate peaks (98 and 145 Hz) | iDON Q64-ERP | 9.55 | 11.95 | 11.18 |
| 5. Different ERP: heavy absorbers on the two high plate peaks (98 and 145 Hz) | iFNO | 4.63 | 13.17 | 10.90 |
| 5. Different ERP: heavy absorbers on the two high plate peaks (98 and 145 Hz) | iLNO | 4.46 | 14.77 | 10.13 |
| 5. Different ERP: heavy absorbers on the two high plate peaks (98 and 145 Hz) | iSTO | 3.79 | 10.86 | 7.08 |
| 5. Different ERP: heavy absorbers on the two high plate peaks (98 and 145 Hz) | iGNO | 6.80 | 9.13 | 7.83 |
| 6. Different ERP: both resonators tuned to 100 Hz, heavy | iDON Q64 | 6.51 | 7.93 | 7.93 |
| 6. Different ERP: both resonators tuned to 100 Hz, heavy | iDON Q64-ERP | 13.00 | 11.53 | 8.77 |
| 6. Different ERP: both resonators tuned to 100 Hz, heavy | iFNO | 3.96 | 12.88 | 9.85 |
| 6. Different ERP: both resonators tuned to 100 Hz, heavy | iLNO | 4.91 | 11.25 | 9.27 |
| 6. Different ERP: both resonators tuned to 100 Hz, heavy | iSTO | 5.28 | 11.77 | 9.84 |
| 6. Different ERP: both resonators tuned to 100 Hz, heavy | iGNO | 7.03 | 11.92 | 9.90 |

Forward: the true design is the input. Inverse: the target ERP is the input; each recovered design is solved with the plate solver and compared with the target.
"Best of 16" picks, with the solver and the target, the best of 16 sampled designs (an oracle choice, not available for a real target). The out-of-range case has no
forward panel because the models encode the design in bounded coordinates (m <= 1 kg), so a 2 kg resonator cannot be given to them.

## Does the inverse use the target? Baselines that ignore it (`inverse_baselines.py`, 200 held-out test targets)

| Method | RMSE (dB) | Median RMSE (dB) | Skill vs mean ERP | Skill vs bare plate |
|---|---|---|---|---|
| mean training ERP | 7.11 | 6.95 | 0.00 | 0.41 |
| bare plate | 9.29 | 9.39 | -0.70 | 0.00 |
| random design (1) | 10.04 | 9.96 | -0.99 | -0.17 |
| random designs (best of 16) | 7.65 | 7.69 | -0.16 | 0.32 |
| iDON Q64 (point estimate) | 6.58 | 6.38 | 0.14 | 0.50 |
| iDON Q64 (best of 16) | 5.09 | 4.76 | 0.49 | 0.70 |
| iDON Q64-ERP (point estimate) | 6.82 | 6.46 | 0.08 | 0.46 |
| iDON Q64-ERP (best of 16) | 5.38 | 5.11 | 0.43 | 0.67 |
| iFNO (point estimate) | 8.06 | 7.78 | -0.28 | 0.25 |
| iFNO (best of 16) | 7.11 | 7.06 | 0.00 | 0.41 |
| iLNO (point estimate) | 8.22 | 7.95 | -0.34 | 0.22 |
| iLNO (best of 16) | 7.21 | 7.32 | -0.03 | 0.40 |
| iSTO (point estimate) | 8.34 | 8.07 | -0.38 | 0.19 |
| iSTO (best of 16) | 7.14 | 7.17 | -0.01 | 0.41 |
| iGNO (point estimate) | 8.28 | 8.18 | -0.35 | 0.20 |
| iGNO (best of 16) | 7.31 | 7.37 | -0.06 | 0.38 |

Skill = 1 - SSE(method) / SSE(baseline): above 0 the method is closer to the target ERP than the baseline. The mean training ERP and the bare plate (resonators too
light to act) do not look at the target. Random designs are drawn from the training set and solved.
