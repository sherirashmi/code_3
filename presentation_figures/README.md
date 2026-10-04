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
