# Vibro-acoustic metamaterials: neural operators for the ERP problem

Forward (resonator configuration → ERP spectrum), inverse (ERP spectrum →
configuration) and invertible (both directions, one set of weights) neural
operators for a plate with 2 or 3 tuned mass resonators, plus displacement-field
operators.

```bash
python main.py        # interactive menu; works from any working directory
```

Every workflow first asks **which dataset** to use (only the datasets that
suit the chosen workflow are listed, with their number of resonators). That
choice sets

1. the **plate-mode basis** used by the physics solver for every reference
   spectrum (prediction plots, solver-scored inverse evaluation) — applied
   automatically when the data are loaded, including in parallel solver
   workers,
2. the **number of resonators** (2 or 3) every model is built for, and
3. the **dataset sub-folder** for checkpoints and plots.

| Tag | Resonators | Configurations | Plate modes | Files |
|---|---:|---:|---|---|
| `10k` | 3 | 10,000 | 15 × 10 | `datasets/erp/3res/10k/` (1 file) |
| `100k` | 3 | 100,000 | 15 × 10 | `datasets/erp/3res/100k/` (2 files) |
| `200k_18modes` | 3 | 200,000 | 6 × 3 | `datasets/erp/3res/200k_18modes/` (4 files) |
| `100k_2res_grid_18modes` | 2 | 100,000 | 6 × 3 | `datasets/erp/2res/100k_2res_grid_18modes/` (4 files) |
| `100k_2res_fixed_m0.2_ft72_18modes` | 2 | 100,000 | 6 × 3 | `datasets/erp/2res/100k_2res_fixed_m0.2_ft72_18modes/` (4 files) |
| `200k_2res_18modes` | 2 | 200,000 | 6 × 3 | `datasets/erp/2res/200k_2res_18modes/` (8 files) |
| `10k_<N>res_fixed_m0.2_ft<f_t>_18modes` | 2 or 3 | 10,000 each | 6 × 3 | `datasets/erp/blocks_<N>res/` (model-bank blocks, f_t = 40 … 100 Hz) |

Every trained model records the `Nx`/`Ny` it was trained with in its
checkpoint (`modal_resolution`), and whenever a model is loaded again that
recorded basis is re-applied for all solver calculations made with it. A model
whose basis differs from the selected dataset is rejected with an error.

(The 18-mode dataset differs from a 150-mode solve of the same configuration
by up to ~25 dB, so mixing bases silently corrupts any solver comparison.)

## `main.py` menu

1. **ERP forward operators** — DON, DNO, FNO, DCO, GNO, STO, SIREN, WNO, NN,
   LNO: train / evaluate / predict, train-all, frequency holdout, error
   breakdown; options: encoder (set / + f_t-sorted), feature scaling
   (z-scored / physical), NN permutation, FNO padding.
2. **ERP inverse models** — MDN, cVAE, Flow, Diffusion, BasisFlow, PadINN,
   Surrogate: train / evaluate / predict, train-all; options: spectrum encoder,
   design parameterisation (full 5 × N / bounded 4 × N). Also the
   **fixed-resonator position models** (MDN / Flow / Diffusion, m and f_t
   fixed) and the **model bank** (one Flow per f_t block + solver check).
3. **Invertible operators** (coupling flow) — iFNO, iDCO, iGNO, iDNO, iWNO,
   iLNO, iSIREN, iSTO: 3-stage training or evaluation; standard /
   recommended / custom options (encoder, design, gate, readout, …).
4. **Invertible DeepONet** — Q8 (strict Q = D), Q64, Q128 and the trunk
   variants Q64-FNO/-DCO/-DNO/-WNO/-LNO/-SIREN: train (stops at epoch 50 if
   the validation loss stopped improving), evaluate, comparison plots.
5. **Displacement field** — the ten architectures on the field dataset.

## Folder layout

Every model family has the same three subfolders: `scripts/` (code),
`models/` (checkpoints), `plots/` (figures, sorted by type).

```
erp_forward/                     forward operators, design -> ERP
  scripts/                       dno.py, fno.py, ..., operator_registry.py, neural_operator_utils.py
  models/<dataset>/<model>.pth
  models/experiments/{frequency_holdout,dco_variants}/   models/legacy/
  plots/models/<dataset>/<MODEL>/  and  plots/models/<dataset>/ALL_MODELS/
  plots/experiments/{frequency_holdout,dco_variants,gno_variants}/  plots/architectures/  plots/legacy/
erp_inverse/                     inverse models, ERP -> design
  scripts/                       mdn.py, flow.py, ..., train_all.py, evaluate.py
  scripts/fixed_resonator/       position-only models (m, f_t fixed) and the model bank
  models/<dataset>/inverse_<model>.pth    models/fixed_resonator/<dataset>/<model>.pth
  plots/models/<dataset>/{<MODEL>,ALL_MODELS}/   plots/fixed_resonator/<dataset>/   plots/experiments/block_bank[_3res]/
erp_invertible/                  coupling-flow invertible operators (iFNO, iDCO, ...)
  scripts/   models/<dataset>/<model>.pth   plots/models/<dataset>/{<MODEL>,ALL_MODELS}/   plots/architectures/
  FORWARD_VS_INVERTIBLE.md       what each invertible model keeps from / changes in its forward model
erp_invertible_deeponet/         invertible DeepONet (Kaltenbach et al.)
  scripts/   models/<dataset>/idon_<variant>.pth   plots/models/<dataset>/{<VARIANT>,ALL_MODELS}/   plots/architectures/
disp_forward/                    forward operators, design -> displacement field
  scripts/   models/<model>.pth   plots/models/<MODEL>/
datasets/
  erp/3res/<tag>/  erp/2res/<tag>/  erp/blocks_2res/  erp/blocks_3res/  displacement/
  scripts/                       dataset generators
dataset_analysis/                dataset statistics, split plots, videos (scripts/ + one folder per dataset)
presentation_figures/            figures for talks (common-form diagrams)
utils/paths.py                   single source of truth for every model/plot path
utils/plot_style.py              thesis LaTeX style for every figure
```

Paths are anchored at the repository root, so every script works from any
working directory. Set `THESIS_OUTPUT_ROOT=/some/folder` to write all models
and plots there instead (same layout; datasets are still read from the
repository, and trained models in the repository remain usable).

`models/legacy/` holds checkpoints trained with older model code that no longer
load into the current architectures.

## Figures

All figures use Computer Modern (LaTeX) typography, LaTeX maths in labels
and 300 dpi output. If a working `latex` + `dvipng` is installed, text is
typeset by real LaTeX; otherwise Matplotlib's built-in Computer Modern
mathtext is used (same look, no TeX needed). Force with `THESIS_USETEX=1/0`.
Matplotlib's LaTeX mode needs `type1cm.sty` (Ubuntu/Debian:
`texlive-latex-extra`, plus `dvipng` and `cm-super`).

## Dataset: 2 resonators on a 14 x 5 position grid (`100k_2res_grid_18modes`)

`datasets/scripts/generate_grid_2res_18modes.py` (about 10 min on 4 cores) creates
100,000 configurations with:

* 2 resonators per configuration;
* positions only on a 14 x 5 equidistant grid: x = 0.05, 0.15, ..., 1.35 m and
  y = 0.05, 0.15, ..., 0.45 m (0.1 m spacing). The two resonators are always on
  different cells, each of the 70*69/2 = 2,415 cell pairs is used 41-42 times,
  and the resonator order is random;
* m in [0.1, 1.0] kg and f_t in [10, 160] Hz by Latin hypercube, with
  k = m(2 pi f_t)^2 derived;
* 6 x 3 = 18 plate modes (set automatically when the dataset is selected).

It is stored as 4 shards of 25,000 configurations and appears as dataset 4 in
every `main.py` menu. Forward and inverse models read the number of
resonators from the data. For the inverse models, choose design
parameterisation 2 (Bounded): 4 numbers per resonator, [m, f_t, x, y], so 8
for this dataset (saved with the `_b12` suffix).

## Resonator encoder: set vs. set + f_t-sorted branch

Training (and evaluate/predict) in `main.py` asks which resonator encoder to use:

1. **Set encoder** (standard): shared per-resonator MLP + mean/max pooling --
   permutation invariant and smooth, but lossy (different resonator sets
   can pool to the same vector).
2. **Set encoder + f_t-sorted branch** (the DCO_sorted design): additionally
   sorts the resonators by ascending f_t and concatenates their features
   (lossless; order fixed by f_t, so still permutation invariant). Both
   branch outputs are concatenated; for DON, DNO, FNO, SIREN, WNO, LNO, iFNO
   and iDCO they are linearly fused back to the original encoder width
   (`SetAndSortedResonatorEncoder`), DCO keeps its own concat-into-lift
   implementation. Saved and plotted as `<MODEL>_sorted` (e.g. `DNO_sorted`),
   so it never overwrites the standard model. Adds roughly 28-45 % parameters.

GNO, STO, NN and iGNO do not use the set encoder and always train in their
standard form. `ResonanceQueryEncoder` is unchanged in both variants.

## Forward-model options and error diagnostics

After the encoder question, forward training (single model, train-all and
frequency holdout) asks only the questions that apply to the selected
architectures. The defaults reproduce the existing models; every option
adds a name suffix, and suffixes combine (e.g. `DNO_sorted_phys`).

| Option | Applies to | What it does | Suffix |
|---|---|---|---|
| Physical feature scaling | all except NN | Modal features `sin(m*pi*x/Lx)`, `sin(n*pi*y/Ly)` (the plate's own mode shapes) instead of `sin(i*pi*x_z)` on z-scored coordinates, and one detuning scale `(f - f_t)/std(f)` | `_phys` |
| Permutation augmentation | NN | Random resonator order in every training batch | `_perm` |
| FFT padding | FNO | `reflect` or `zero` instead of `replicate` | `_reflect`, `_zpad` |

Why the physical scaling: the z-scored x and y are centred on the plate
middle, so every standard feature `sin(i*pi*x_z)` is odd about the centre.
The odd-index plate modes `sin(m*pi*x/Lx)` (m = 1, 3, 5, ...) are even about
it, so none of the 10 standard harmonics can represent them. A least-squares
fit of each plate mode from those harmonics gives R^2 = 0.001 / 0.03 / 0.002
for m = 1 / 2 / 3 along x, and similar along y. The models still learn the
mode shapes from the raw x, y inputs; the `_phys` variant hands them over
directly. The detuning change is small: with the standard scaling the
offset is at most 0.24 Hz, under half a frequency step.

The scaling factors (x and y mean/std over Lx, Ly; f_t over f
statistics) are taken from the training data's normalisation and stored as
buffers in the checkpoint, so evaluate/predict reproduce them exactly.

**Error diagnostics.** Every evaluation now also prints the RMSE in the
lowest / interior / highest 5 % of the frequency axis, at vs. off the true
resonance peaks, and on the 10 % of test configurations with the closest
pair of tuning frequencies or a resonator nearest a plate edge. The same
columns go into the train-all comparison table, plus an
`ALL_MODELS/error_breakdown_bars.png`. To re-score already trained models
without retraining, use forward method 3 in `main.py`, or run
`python -m erp_forward.scripts.diagnose 100k`.

100k results (`erp_forward/plots/models/100k/ALL_MODELS/error_breakdown.txt`, RMSE in dB):

| Model | All | Low 5 % f | Interior | High 5 % f | At peaks | Off peaks | Close f_t | Near edge |
|---|---|---|---|---|---|---|---|---|
| DNO | 2.69 | 0.25 | 2.75 | 2.84 | 11.74 | 2.06 | 2.81 | 2.58 |
| LNO | 2.86 | 0.37 | 2.93 | 2.95 | 12.48 | 2.18 | 2.99 | 2.71 |
| STO | 2.94 | 0.42 | 3.02 | 2.99 | 13.39 | 2.17 | 3.13 | 2.76 |
| SIREN | 2.98 | 0.43 | 3.05 | 3.19 | 12.78 | 2.31 | 3.10 | 2.88 |
| FNO | 3.11 | 0.94 | 3.15 | 3.71 | 13.67 | 2.37 | 3.26 | 2.95 |
| WNO | 3.30 | 1.88 | 3.33 | 3.72 | 14.34 | 2.53 | 3.42 | 3.21 |
| DCO | 3.68 | 0.35 | 3.76 | 3.98 | 7.60 | 3.53 | 3.92 | 3.50 |
| GNO | 4.41 | 0.53 | 4.51 | 4.63 | 7.86 | 4.29 | 4.62 | 4.10 |
| DON | 5.21 | 0.53 | 5.39 | 4.40 | 20.39 | 4.27 | 5.20 | 4.91 |

Key observations:

* Error at the resonance peaks is 4-5 times the overall RMSE. DCO and GNO
  are clearly best at the peaks despite ranking low overall.
* FNO and WNO have a visible low-frequency edge error (2-5 times the
  others), consistent with the FFT/wavelet boundary.
* The close-f_t stratum is only 3-5 % harder; near-edge resonators are
  slightly easier (they couple weakly to the plate).

**LNO poles.** LNO's pole/residue terms are a learned rational basis on the
normalised frequency axis that feeds a nonlinear dB-output head. They are
not identified physical poles, so do not report them as natural
frequencies or damping ratios.

## Inverse-model options and evaluation

Inverse training / evaluate / predict in `main.py` ask for two options
(saved as part of the model name, so variants never overwrite each other):

| Option | Choices | Name suffix |
|---|---|---|
| Spectrum encoder | **pooled** (conv + global mean/max; all existing models) or **positional** (normalised-frequency input channel + 16 ordered frequency bins, flattened, so peak *positions* are kept) | `_pos` |
| Design space | **full 15-D** `[m,k,f_t,x,y]` z-scored (existing models) or **bounded 12-D** `[m,f_t,x,y]`, logit-bounded to the generation ranges, `k = m(2πf_t)²` derived (`erp_inverse/scripts/design_space.py`) | `_b12` |

With the bounded 12-D space every model output decodes to a valid,
consistent design (no clipping), and densities are exact: they are
converted to the physical `(m, f_t, x, y)` space with the transform's
log-Jacobian. In the 15-D space predicted designs are projected
(`k` re-derived, bounds clipped) and **re-scored** at the projected design,
so the reported log p belongs to the design that is evaluated.
SurrogateInverse derives `k` before querying its frozen forward surrogates.

**Evaluation** (`evaluate.py`) reports three selection rules per model, each
scored with the actual solver:

* **random** -- the first i.i.d. sample (no selection);
* **own** -- the model's own target-blind rule: highest log p (MDN, Flow,
  BasisFlow exact; cVAE via an importance-sampled marginal, 64 samples),
  the point estimate for Surrogate, n/a for Diffusion/PadINN;
* **oracle** -- best of N by solver error against the target, the same rule
  for every model (uses the target, so it is an upper bound).

Design-parameter recovery (`evaluate_design.py`) uses the target-blind
selection only and sorts predicted resonators by predicted `f_t` before the
slot-by-slot comparison. The "%" shown next to sampled designs is a softmax
over the drawn samples -- a relative ranking, not a calibrated probability.
Validation losses use fixed randomness (and the cVAE its final KL weight),
so best-epoch selection is not Monte-Carlo luck.

## Running on a GPU (e.g. Google Colab)

The code uses CUDA automatically when available (`utils/support.py`).
In Colab: *Runtime -> Change runtime type -> T4 GPU*, then in cells:

```python
!git clone -b ai_code https://github.com/sherirashmi/code_3.git   # private repo: use a token URL
%cd code_3
!pip -q install scipy matplotlib     # torch with CUDA is preinstalled on Colab
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
!python main.py                      # answer the prompts in the input box
```

**Interruption-safe runs:** `python cloud_train.py --out <persistent folder> --models DNO,GNO,STO --physical`
links `erp_forward/models` and `erp_forward/plots` into the persistent folder
(Google Drive on Colab, `/kaggle/working/...` on Kaggle). Every 5 epochs each
model's full state is saved as `<model>.resume.pt` plus
`<model>_progress.csv/.png`. Re-running the same command after a disconnect
skips finished models and continues the interrupted one from its last save.
`python cloud_train.py --help` lists all options.

Colab sessions end after ~12 h or when idle, and their disk is wiped:
copy results out when a run finishes, e.g. mount Drive
(`from google.colab import drive; drive.mount('/content/drive')`) and
`!cp -r erp_*_operators/models erp_*_operators/plots /content/drive/MyDrive/thesis_runs/`.
A GPU speeds up training; the solver-scored evaluation runs on the CPU cores.

## Training details

### Forward operators (`erp_forward/scripts/neural_operator_utils.py`)

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

### Invertible operators (`erp_invertible/scripts/train.py`) — staged training

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

**Configurations** (asked in `main.py`; non-default choices are part of the
model name): *Standard* (paper-style, the existing checkpoints) or
*Recommended* = all of

| Option | Suffix | What it does |
|---|---|---|
| bounded 12-D design | `_b12` | predict `[m, f_t, x, y]`, logit-bounded; `k` derived and the z-scored configuration rebuilt before the forward lift |
| bounded gate | `_bg` | `S = exp(c·tanh(a·L))`, `a = 0` at init -> every block starts as the identity, scale in `[e^-c, e^c]` (paper: `softplus(L)`, 0.69 at init, unbounded) |
| binned readout | `_bin` | design readout gets 16 ordered frequency bins besides mean/max pooling |
| FFT padding | `_pad` | iFNO only: zero-pad the non-periodic frequency axis by 8 points before the FFT |
| cycle + alignment | `_cyc` | `0.1·MSE(x̂(ŷ(x)), x) + 0.1·‖P'(y) − F(P(x))‖²/‖F(P(x))‖²` in stages 1 and 3 |
| stage-2 on estimates | `_s2e` | VAE pretrained on stage-1 inverse estimates (its stage-3 input) instead of true designs |

Only the coupling stack is exactly invertible (between latent tensors);
lifts and readouts are learned and lossy, so these are bidirectional models
with an invertible core, without an exact density over designs. Every
stage-1/3 epoch logs the core's round-trip error, latent-norm growth, gate
range and forward/inverse latent misalignment (also stored in the
checkpoint history).

Inverse evaluation: 8 samples per target are projected onto physically
consistent designs (k = m(2πf_t)², bounds clipped), and scored with the actual coupled solver under: a random sample; the
sample selected with the model's own forward direction (target-informed,
no solver); the point estimate; and the oracle best-of-N by solver. Predicted
resonators are sorted by f_t before parameter recovery.

### Inverse models (`erp_inverse/scripts/`)

MDN, cVAE, conditional flow, conditional diffusion, BasisFlow, PadINN and
the surrogate-in-the-loop inverse (frozen DCO + GNO forward operators; uses
the forward checkpoints of the same dataset when they exist). Adam, cosine
schedule to 1 % LR, gradient clipping 5, best-validation restore; per-model
LR/epochs in `erp_inverse/scripts/registry.py`.
