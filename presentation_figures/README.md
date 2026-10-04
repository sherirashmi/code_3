# Presentation figures

Architecture figures for slides (white background, Computer Modern text; PNG, PDF and SVG each).

| Figure | Model drawn | Source of every number |
|---|---|---|
| `forward_operator` | Deep Cat Operator (final variant: set + f_t-sorted encoder, plate-mode features) | `models/GENERAL/100k/dco_sorted_phys.pth` (model_config, training_config), `erp_forward_operators/dco.py`, `neural_operator_utils.py` |
| `inverse_model` | Conditional normalizing flow `Flow_b12` | `erp_inverse_operators/models/100k_2res_grid_18modes/inverse_flow_b12.pth`, `flow.py`, `common.py`, `design_space.py`, `registry.py` |
| `invertible_operator` | iFNO with `RECOMMENDED_OPTIONS` (sorted encoder, plate-mode features, bounded design and gate, binned readout, padded FFT) | parameter counts of `IFNO(design_dim=8, num_res=2)` built with those options; `ifno.py`, `coupling.py`, `common.py`, `train.py` |

`inverse_example.npz` holds the real held-out target and the 8 designs sampled from the trained Flow
(`make_inverse_example.py`, run from the repository root); the candidate plate in `inverse_model` is drawn from it.

Regenerate (from the repository root, with `PYTHONPATH=.`):
`python presentation_figures/forward_operator_figure.py` (same for `inverse_model_figure.py`, `invertible_operator_figure.py`).
