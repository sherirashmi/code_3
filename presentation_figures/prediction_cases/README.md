# True vs predicted: the final trained models on unseen cases

Inference only (nothing was retrained). Ground truth is always the plate solver. Scripts: `forward_cases.py`, `inverse_cases.py` (shared helpers in `common.py`).
Exact checkpoints, designs and per-model numbers: `forward_novelty.txt`, `inverse_novelty.txt`, `forward_cases.csv`, `inverse_cases.csv`.

## Checkpoints used

Forward operators, 3 resonators, 100k dataset (`erp_forward/models/100k/`): DON `don_sorted_phys.pth`, DNO `dno_sorted_phys.pth`, DCO `dco_sorted_phys.pth`,
FNO `fno_sorted_phys.pth`, WNO `wno_sorted_phys.pth`, LNO `lno_sorted_phys.pth`, GNO `gno_phys.pth`, STO `sto_phys.pth`, SIREN `siren_sorted_phys.pth`, NN `nn_perm.pth`.

Invertible models, 2 resonators, 200k dataset: iDON Q64 `erp_invertible_deeponet/models/200k_2res_18modes/idon_q64.pth`, iDON Q64-ERP `.../idon_q64-erp.pth`;
iFNO `erp_invertible/models/200k_2res_18modes/ifno_sortenc_phys_b12_bg_bin_pad_cyc_s2e.pth`, iLNO `ilno_sortenc_phys_b12_bg_bin_cyc_s2e.pth`,
iSTO `isto_phys_b12_bg_bin_cyc_s2e.pth`, iGNO `igno_phys_b12_bg_bin_cyc_s2e.pth`.

## Forward ERP prediction, RMSE vs solver (dB), figures `forward_case_<n>.png`

| Case | DON | DNO | DCO | FNO | WNO | LNO | GNO | STO | SIREN | NN |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Test-split configuration 3 (held out) | 5.65 | 3.48 | 3.59 | 4.49 | 4.26 | 4.57 | 3.86 | 3.64 | 3.64 | 7.99 |
| 2. New, inside the training ranges | 7.31 | 1.74 | 1.97 | 2.80 | 3.65 | 2.14 | 3.65 | 3.50 | 2.33 | 7.72 |
| 3. New, all resonators at 72 Hz on one line | 8.23 | 3.72 | 2.55 | 4.70 | 4.94 | 5.26 | 5.84 | 7.84 | 3.22 | 9.63 |
| 4. Outside the training ranges (heavy resonators 1.5-2.5 kg) | 12.21 | 10.91 | 8.60 | 12.99 | 10.64 | 8.01 | 14.32 | 14.80 | 10.52 | 11.40 |

## Invertible models, figures `inverse_case_<n>_forward.png` (design -> ERP) and `inverse_case_<n>_inverse.png` (ERP -> design, checked with the solver)

| Case | Model | Forward RMSE (dB) | Inverse: point estimate (dB) | Inverse: best of 16 (dB) |
|---|---|---|---|---|
| 1 test-split | iDON Q64 | 2.02 | 6.41 | 6.41 |
| 1 test-split | iDON Q64-ERP | 3.56 | 5.05 | 5.04 |
| 1 test-split | iFNO | 1.52 | 8.63 | 7.57 |
| 1 test-split | iLNO | 1.43 | 10.71 | 9.19 |
| 1 test-split | iSTO | 1.53 | 9.50 | 9.52 |
| 1 test-split | iGNO | 2.79 | 11.27 | 7.38 |
| 2 new, in range | iDON Q64 | 2.85 | 8.04 | 7.27 |
| 2 new, in range | iDON Q64-ERP | 3.80 | 3.21 | 3.21 |
| 2 new, in range | iFNO | 2.94 | 6.81 | 4.71 |
| 2 new, in range | iLNO | 2.25 | 8.92 | 5.65 |
| 2 new, in range | iSTO | 2.63 | 8.64 | 7.56 |
| 2 new, in range | iGNO | 3.43 | 7.32 | 5.70 |
| 3 outside range | iDON Q64 | n/a | 10.35 | 9.07 |
| 3 outside range | iDON Q64-ERP | n/a | 13.01 | 9.92 |
| 3 outside range | iFNO | n/a | 10.78 | 9.40 |
| 3 outside range | iLNO | n/a | 13.57 | 8.96 |
| 3 outside range | iSTO | n/a | 10.60 | 9.45 |
| 3 outside range | iGNO | n/a | 9.51 | 8.89 |

Forward: the true design is the input. Inverse: the target ERP is the input; each recovered design is solved with the plate solver and compared with the target.
"Best of 16" picks, with the solver and the target, the best of 16 sampled designs (an oracle choice, not available for a real target). The out-of-range case has no
forward panel because the models encode the design in bounded coordinates (m <= 1 kg), so a 2 kg resonator cannot be given to them.
