# Vibro-acoustic metamaterials: neural operators for the ERP problem

Forward (resonator configuration → ERP spectrum), inverse (ERP spectrum →
configuration) and invertible (both directions, one set of weights) neural
operators for a plate with 3 tuned mass resonators, plus displacement-field
operators.

```bash
python main.py        # interactive menu; works from any working directory
```

Every ERP workflow first asks **which dataset** to use. That choice sets

1. the **plate-mode basis** used by the physics solver for every reference
   spectrum (prediction plots, solver-scored inverse evaluation) — applied
   automatically when the data are loaded, including in parallel solver
   workers, and
2. the **dataset sub-folder** for checkpoints and plots.

| Tag            | Files                                        | Configurations | Plate modes        |
|----------------|----------------------------------------------|---------------:|--------------------|
| `10k`          | `datasets/dataset_erp_ft.pth`                 | 10 000         | 15 × 10 = 150      |
| `100k`         | `datasets/dataset_erp_ft_100k_part{1,2}.pth`  | 100 000        | 15 × 10 = 150      |
| `200k_18modes` | `datasets/dataset_erp_ft_200k_18_modes_part{1..4}.pth` | 200 000 | 6 × 3 = 18   |

Choosing the dataset immediately sets the solver's `Nx`/`Ny` (15/10 for the
150-mode datasets, 6/3 for the 18-mode one). Every trained model records the
`Nx`/`Ny` it was trained with in its checkpoint (`modal_resolution`), and
whenever a model is loaded again (evaluate, predict, inverse sampling,
invertible evaluation) that recorded basis is re-applied for all solver
calculations made with it. A model whose basis differs from the selected
dataset is rejected with an error. Checkpoints saved before this field
existed fall back to their dataset's basis.

(The 18-mode dataset differs from a 150-mode solve of the same configuration
by up to ~25 dB, so mixing bases silently corrupts any solver comparison.)

## Folder layout

```
erp_forward_operators/
  models/GENERAL/<dataset>/<model>.pth          main.py: ERP → Forward → General
  models/FREQ_HOLDOUT/<dataset>/<model>.pth     main.py: ERP → Forward → Frequency holdout
  models/DCO_VARIANTS/  models/LEGACY/          archived experiments / outdated checkpoints
  plots/GENERAL/<dataset>/<MODEL>/              loss_curve, erp_spectrum_test_config_01..05,
                                                prediction_vs_ground_truth, prediction_spectrum
  plots/GENERAL/<dataset>/ALL_MODELS/           all_training_loss, all_validation_loss,
                                                all_models_loss, box plots, bar charts, comparison_table.txt
  plots/FREQ_HOLDOUT/<dataset>/{<MODEL>,ALL_MODELS}/
  plots/DCO_VARIANTS/  plots/GNO_VARIANTS/  plots/LEGACY/
erp_inverse_operators/
  models/<dataset>/inverse_<model>.pth
  plots/<dataset>/<MODEL>/   (loss_curve, single-model evaluation)
  plots/<dataset>/ALL_MODELS/ (all_models_loss, validation_reconstructions, stats, recovery)
erp_invertible_operators/          (iFNO, iDCO, iGNO)
  models/<dataset>/<model>.pth
  plots/<dataset>/<MODEL>/   (loss_curve, stage_losses, forward plots, inverse plots, metrics.json)
  plots/<dataset>/ALL_MODELS/
utils/paths.py        single source of truth for all of the above
utils/plot_style.py   thesis LaTeX style for every figure
```

`LEGACY/nn_100k_before_plain_mlp.pth`, `dco_high.pth`, `dco_low.pth`,
`dno_high.pth`, … were trained with older model code and no longer load into
the current architectures; retrain them if needed.

## Figures

All figures use Computer Modern (LaTeX) typography, LaTeX maths in labels
and 300 dpi output. If a working `latex` + `dvipng` is installed, text is
typeset by real LaTeX; otherwise Matplotlib's built-in Computer Modern
mathtext is used (same look, no TeX needed). Force with `THESIS_USETEX=1/0`.
Matplotlib's LaTeX mode needs `type1cm.sty` (Ubuntu/Debian:
`texlive-latex-extra`, plus `dvipng` and `cm-super`).

## Training details

### Forward operators (`erp_forward_operators/neural_operator_utils.py`)

Loss on z-scored ERP (`erp_spectrum_loss`):

$$\mathcal{L} = \mathrm{MSE}(\hat y, y) + 0.5\,\mathrm{MSE}(\Delta\hat y, \Delta y)
 + 0.05\,\tfrac{1}{B}\sum_b \sum_{f\in\mathcal{P}_b}(\hat y_{bf}-y_{bf})^2$$

(Δ = first difference along frequency; 𝒫_b = all local maxima of the true
spectrum b, window 7.)

* AdamW, weight decay 1e-4, cosine LR schedule to 1e-6, gradient clipping 5,
  best-validation weights restored; optional L-BFGS polish (off by default).
* Split 80/10/10 by configuration; seed 727; inputs/targets z-scored on the
  training split.
* Every new checkpoint stores its `training_config` and loss `history`.

| Model | Epochs | LR | Params | Architecture hyperparameters |
|---|---:|---:|---:|---|
| DON | 200 | 5e-4 | 113,885 | hidden_dim=45, context_dim=71, basis_dim=113, num_terms=4, refine_width=28, tanh |
| DNO | 200 | 5e-4 | 117,030 | hidden_dim=54, context_dim=57, frequency_dim=28, query_dim=28, depth=4, silu |
| FNO | 200 | 5e-4 | 119,560 | width=23, modes=35, depth=4, config_hidden=70, query_dim=26, padding=8, dropout=0.1, gelu |
| DCO | 200 | 5e-4 | 117,932 | hidden_dim=67, branch_dim=56, trunk_dim=56, query_dim=28, depth=4, silu |
| GNO | 200 | 5e-4 | 112,230 | width=60, depth=3, frequency_dim=28, modal_harmonics=4, dropout=0.1, silu |
| STO | 200 | 5e-4 | 110,193 | width=56, heads=4, depth=2, ff_dim=144, modal_harmonics=4, dropout=0.1, gelu |
| SIREN | 200 | 2e-4 | 118,147 | context_dim=57, query_dim=14, hidden_dim=75, config_hidden=57, query_hidden=28, depth=4, ω₀=20 |
| WNO | 200 | 5e-4 | 117,638 | width=30, depth=4, levels=3, config_hidden=57, query_dim=22, dropout=0.1, gelu |
| NN | 200 | 5e-4 | 112,925 | hidden_dim=148, depth=6, dropout=0.1, relu |
| LNO | 200 | 5e-4 | 112,340 | width=90, num_poles=13, config_hidden=64, pole_hidden=64, query_dim=25, dropout=0.1, silu |

### Invertible operators (`erp_invertible_operators/train.py`) — staged training

Yes, staged: three stages after Long et al. (arXiv:2402.11722, Sec. 3.3).
All terms are MSEs on z-scored quantities (design vector sorted by f_t).

| Stage | Trains | Objective | Epochs (default) | LR |
|---|---|---|---:|---:|
| 1 | coupling blocks + lifts/projections (not the VAE) | J_FWD + J_INV + J_{P,Q'} + J_{P',Q} | 20 | 5e-4 |
| 2 | β-VAE only, on true designs | MSE(recon) + β·KL, β warmed 0 → 0.05 over 8 epochs | 25 | 1e-3 |
| 3 | everything | J_FWD + J_{P,Q'} + J_{P',Q} + J_β-VAE + w_inv·J_INV (β = 0.05, w_inv = 1) | 15 | 3e-4 |

Adam, cosine LR to 1 % of the stage LR, gradient clipping 5, best-validation
weights restored after each stage, batch size 128. `w_inv = 0` reproduces
the paper's eq. 11 exactly. All values can be changed from `main.py`.

| Model | Params | Hyperparameters |
|---|---:|---|
| iFNO | 194,827 | width=24, num_blocks=4, modes=48, config_hidden=80, query_dim=32, z_dim=6, vae_hidden=64, τ=1, gelu |
| iDCO | 81,491 | width=24, num_blocks=4, branch_dim=56, trunk_dim=28, query_dim=32, z_dim=6, vae_hidden=64, τ=1, silu |
| iGNO | 59,272 | width=24, num_blocks=4, depth=3, frequency_dim=28, modal_harmonics=4, dropout=0.1, z_dim=6, vae_hidden=64, τ=1, silu |

Inverse evaluation: 8 samples per target are projected onto physically
consistent designs (k = m(2πf_t)², bounds clipped), one is selected with the
model's own forward direction (no solver, no ground truth), and that design
is scored with the actual coupled solver. Point-estimate and oracle
best-of-N scores are reported alongside.

### Inverse models (`erp_inverse_operators/`)

MDN, cVAE, conditional flow, conditional diffusion, BasisFlow, PadINN and
the surrogate-in-the-loop inverse (frozen DCO + GNO forward operators; uses
the forward checkpoints of the same dataset when they exist). Adam, cosine
schedule to 1 % LR, gradient clipping 5, best-validation restore; per-model
LR/epochs in `erp_inverse_operators/registry.py`.
