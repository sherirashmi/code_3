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
