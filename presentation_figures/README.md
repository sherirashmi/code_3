# Presentation figures

Architecture figures for slides (white background, Computer Modern text; PNG, PDF and SVG each).

| Figure | Model drawn | Source of every number |
|---|---|---|
| `forward_operator` | Deep Cat Operator (final variant: set + f_t-sorted encoder, plate-mode features) | `erp_forward/models/100k/dco_sorted_phys.pth` (model_config, training_config), `erp_forward/scripts/dco.py`, `neural_operator_utils.py` |
| `inverse_model` | Conditional normalizing flow `Flow_b12` | `erp_inverse/models/100k_2res_grid_18modes/inverse_flow_b12.pth`, `flow.py`, `common.py`, `design_space.py`, `registry.py` |
| `invertible_operator` | iFNO with `RECOMMENDED_OPTIONS` (sorted encoder, plate-mode features, bounded design and gate, binned readout, padded FFT) | parameter counts of `IFNO(design_dim=8, num_res=2)` built with those options; `ifno.py`, `coupling.py`, `common.py`, `train.py` |

`inverse_example.npz` holds the real held-out target and the 8 designs sampled from the trained Flow
(`make_inverse_example.py`, run from the repository root); the candidate plate in `inverse_model` is drawn from it.

Regenerate (from the repository root, with `PYTHONPATH=.`):
`python presentation_figures/forward_operator_figure.py` (same for `inverse_model_figure.py`, `invertible_operator_figure.py`).

## Common-form figures (all variants of a family in one template)

The block that changes between variants gets a general name and a dashed orange frame; every variant has a card
with its own structure and parameter count.

| Figure | Family | General (changing) block | Variants on the cards |
|---|---|---|---|
| `forward_operators_common` | forward operators | Operator core | NN, DON, DNO, FNO, DCO, GNO, STO, SIREN, WNO, LNO |
| `inverse_models_common` | inverse models | Conditional generative head | MDN, cVAE, Flow, Diffusion, BasisFlow, PadINN |
| `invertible_deeponet_common` | Invertible DeepONet | Trunk (basis functions of the frequency) | MLP, FNO, DCO, DNO, WNO, LNO, SIREN (Q = 64); size variants Q8 / Q64 / Q128 |
| `invertible_coupling_common` | invertible coupling-flow operators | Lift P and gate network L | iFNO, iDCO, iGNO, iDNO, iWNO, iLNO, iSIREN, iSTO |

Sources: parameter counts and training settings from the committed checkpoints (`erp_forward/models/100k`,
`erp_inverse/models/100k_2res_grid_18modes`, `erp_invertible_deeponet/models/200k_2res_18modes`) and from
models built with the code's recommended options (coupling-flow operators); structures from each class's `forward()`.
Scripts: `*_common_figure.py`, shared helpers in `common_forms_kit.py`.

## Minimal block diagrams (light colours, plate and ERP icons)

| Figure | Content |
|---|---|
| `forward_model_block_diagram`, `forward_core_architectures` | forward model and the core operation of each architecture |
| `idon_block_diagram` | Invertible DeepONet (Q8, Q64): RealNVP and basis blocks, forward and inverse arrows |
| `invertible_gate_networks` | the gate network L of each invertible operator (list, same style as the forward core list) |
| `ifno_family_block_diagram` | invertible coupling-flow operators (iFNO and variants): lift, coupling stack with gate L, readout |

The icons use one real 2-resonator example (`inverse_example.npz`).  Scripts: `forward_model_block_diagram.py`,
`forward_core_list_figure.py`, `idon_ifno_block_diagrams.py`.

## Plate model and equations

| Figure | Content |
|---|---|
| `plate_three_resonators` | simply supported plate with the driving force and three sprung-mass resonators (positions are illustrative) |
| `plate_equation_step1_governing_equations` ... `plate_equation_step5_velocity_power_erp` | the equations from the plate and resonator PDEs, through the modal expansion, coupled system and harmonic solve, to the ERP, one image per step, same font size |

Equations follow `utils/physics.py` and `utils/solver.py` (resonator damping c = 1 N s/m, real plate modal frequencies).
Scripts: `plate_schematic_figure.py`, `plate_equation_steps_figure.py`.

## Plate demo: field, velocity and ERP sweep

| Figure | Content |
|---|---|
| `plate_demo_1_displacement` | displacement magnitude (mm) over the plate at 103 Hz (1 N force, three resonators at the schematic positions, $m$ = 0.5, 0.8, 0.6 kg, $f_t$ = 45, 90, 130 Hz) |
| `plate_demo_2_velocity` | velocity magnitude $\omega|w|$ (mm/s, same length unit) from that displacement field, in a different colour scale (viridis) |
| `plate_demo_3_erp_sweep` | ERP over 10-160 Hz with the frequency of the field plots marked |

Script: `plate_demo_three_pictures.py` (optional argument: the frequency in Hz; solver `utils/solver.py`, 150 modes).

## Training loss

| Figure | Content |
|---|---|
| `loss_function` | forward-operator loss: MSE + 0.5 slope MSE + 0.05 peak squared error (`erp_spectrum_loss`, normalised ERP; peaks = local maxima of the true spectrum) |

Script: `loss_equation_figure.py`. The committed invertible DeepONet Q8/Q64 checkpoints were trained with plain MSE.

| `idon_loss_function` | Invertible DeepONet loss with the ERP option (Q64-ERP): MSE + 0.5 slope + peak term from 80 % of the epochs + inverse and latent Huber terms; script `idon_loss_equation_figure.py` |

## Peak-term schedule (DCO, 100k test split)

| Figure | Content |
|---|---|
| `peak_schedule_comparison` | DCO with the peak term from the first epoch (`erp_forward/models/100k/dco.pth`) vs from 80 % of the epochs (`dco_staged.pth`) on the first 8 test configurations; metrics over all 10,000 test spectra in `peak_schedule_metrics.json` |

Script: `peak_schedule_comparison.py`. The checkpoints were identified by reproducing the errors of the repo's `peak_term_evolution` plots on the first test configuration. The no-peak DCO and GNO models of those plots are not in the repo.

| `peak_schedule_config4_1_mse_slope`, `..._2_peak_term_added`, `..._3_peak_term_from_80_percent` | DCO on test configuration 4 (100k split), one image per model: no peak term (`legacy/dco.pth`), peak term from the first epoch (`100k/dco.pth`), staged peak term (`100k/dco_sorted_phys.pth`, 200 epochs, also sorted resonators and physical features); script `peak_schedule_config4.py` |

| `idon_base_loss_function` | loss of the base Invertible DeepONet (plain MSE variants Q8, Q64): MSE + warm-up-weighted inverse Huber term + latent Huber term; script `idon_base_loss_equation_figure.py` |

| `idon_inverse_q64_vs_q64erp` | inverse design example (test configuration 3) of the Invertible DeepONet Q64 and Q64-ERP; the existing evaluation images with the figure title cropped away and the title of the left plot replaced by each model's loss equation (script `idon_inverse_example_titles.py`) |

| `idon_base_loss_simple` | the base Invertible DeepONet loss in plain words (ERP error + w x design recovery error + 0.1 x padding error); script `idon_base_loss_simple_figure.py` |

| `idon_base_loss_compact` | the base Invertible DeepONet loss as four short equations with simple symbols (L_ERP, L_design, L_pad, w) with the meaning of a, z, e; script `idon_loss_compact_figure.py` |

| `idon_erp_loss_compact` | the same with the slope and peak terms added (Q64-ERP), with the meaning of every symbol; script `idon_loss_compact_figure.py` |

| `idon_inverse_q64erp` | the Q64-ERP panel of the inverse example alone (test configuration 3), with no title above the plots (same script `idon_inverse_example_titles.py`) |

| `forward_metrics_table` | forward operators: core operation plus RMSE, R^2, correlation (Pearson r over all points) and RMSE at the true resonance peaks of the latest 200-epoch checkpoints in `erp_forward/models/100k` (values read from `erp_forward/plots/models/100k/ALL_MODELS/forward_models_metrics.csv`); script `forward_metrics_table_figure.py` |

| `ifno_loss_function` | training loss of the invertible coupling-flow operators (iFNO and variants) in its three stages, with simple symbols and the meaning of each term (`erp_invertible/scripts/common.py`: `stage1_loss`, `stage2_loss`, `stage3_loss`); script `ifno_loss_figure.py` |

| `invertible_models_one_plot` | iFNO, iLNO, iSTO and iGNO (`erp_invertible/models/200k_2res_18modes`) on test configuration 3: forward ERP prediction of all four in one plot, the plate with the true resonators and each model's inverse-predicted design (point estimate), and a table of the predicted mass and tuning frequency; script `invertible_models_one_plot.py` |

| `invertible_models_idon_style` | inverse design of test configuration 3 by iFNO, iLNO, iSTO and iGNO in the layout of the iDON example: ERP of 16 sampled designs, the point estimate and the best of 16 checked with the solver, plus the plate with true and predicted resonators; script `invertible_models_idon_style.py` |

| `invertible_best16_iFNO`, `..._iLNO`, `..._iSTO`, `..._iGNO`, `invertible_best16_plate` | best of 16 sampled designs on test configuration 3: one ERP graph per model (solver-checked design against the target ERP, true tuning frequencies dashed) and one plate with the true resonators and the best designs of all four models; script `invertible_models_best_of_16_separate.py` |

| `forward_general_block_diagram`, `inverse_general_block_diagram`, `invertible_general_block_diagram` | general top-to-bottom block diagrams (input, architecture, output): forward is one design to one ERP, inverse is one ERP to several possible designs (many designs share one ERP); invertible is one architecture used in both directions (design to ERP and back); script `general_block_diagrams.py` |

| `erp_frequency_band_plot` | ERP distribution at each of the 301 frequency points over the 100k three-resonator dataset (panel (a) of `dataset_analysis/100k/plots/erp_frequency_boxplot.png` alone, without the "(a)"), drawn from `dataset_analysis/100k/stats/erp_frequency_boxplot.csv`; script `erp_frequency_band_plot.py` |

| `erp_frequency_band_plot_200k_2res_18modes` | the same plot for the 200k two-resonator dataset (18 modes); statistics computed with `dataset_analysis/scripts/plot_erp_frequency_boxplot.py 200k_2res_18modes` (written to `dataset_analysis/200k_2res_18modes/`); `python presentation_figures/erp_frequency_band_plot.py 200k_2res_18modes` |

| `forward_loss_curves` | training and validation loss of the forward operators (100k, 200 epochs, histories from the checkpoints in `erp_forward/models/100k`): two panels without titles (validation curves smoothed with a 9-epoch moving average), y axes "Training loss" / "Validation loss", one shared legend and the loss equation underneath; script `forward_loss_curves_figure.py` |

| `plate_equation_kirchhoff_love` | the Kirchhoff-Love thin plate equation D nabla^4 w + rho h d^2w/dt^2 = q, the expansion of nabla^4 and D, with the meaning of each symbol; script `kirchhoff_love_equation_figure.py` |

| `plate_equation_short` (and `_transparent`) | the Kirchhoff-Love plate equation in short form, D nabla^4 w + rho h w'' = f, bare like the Helmholtz equation on the approaches slide; script `kirchhoff_love_short_figure.py` |

| `forward_metrics_table_slide.pptx` | the forward-operator table as a one-slide PowerPoint of native, editable shapes (rounded blocks and text, no picture); generated by `forward_metrics_table_pptx.js` (needs `npm install pptxgenjs`; values from `forward_models_metrics.csv`) |

| `block_diagrams_slides.pptx` | the forward-model block diagram, the core-architecture list, the general forward / inverse / invertible diagrams, the Invertible DeepONet diagram and the iFNO family diagram as one 5-slide PowerPoint of native, editable shapes (blocks, arrows, plate icons, ERP curve as a freeform, text as text); generated by `diagrams_pptx.js` (needs `npm install pptxgenjs`; ERP curve points in `erp_curve.json`) |

| `activation_functions` | curves of the activation functions used in the forward operators (`erp_forward/scripts`, `DEFAULT_MODEL_CONFIG`): ReLU (NN), GELU (FNO, WNO, STO), SiLU (DNO, DCO, LNO, GNO, refinement conv), Tanh (DON, shared encoders, FiLM gates), sine with omega_0 = 20 (SIREN), softplus (LNO poles), softmax (attention and DON term weights as bars) and a comparison of the ReLU family; script `activation_functions_figure.py` |

| `forward_metrics_table.xlsx` | the forward-operator table as an Excel sheet (same rows, core operations and group colours; best value green and worse-than-halfway red as conditional formats that follow the numbers); script `forward_metrics_table_xlsx.py` (needs `openpyxl`) |

| `prediction_cases/` | true vs predicted ERP of the final trained models on unseen cases (inference only): `forward_case_1..4.png` (10 forward operators; test-split configuration, two new in-range configurations, one outside the training ranges), `inverse_case_1..3_forward.png` / `_inverse.png` (iDON Q64, Q64-ERP, iFNO, iLNO, iSTO, iGNO: forward ERP, and ERP -> design checked with the solver, with the plate); the checkpoints used, the numbers and the distance of each case to the training set are in `prediction_cases/README.md`, `*_novelty.txt` and `*.csv`; scripts `forward_cases.py`, `inverse_cases.py` |
