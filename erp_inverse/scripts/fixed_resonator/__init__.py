"""Position-only inverse models (MDN, Flow, Diffusion) for datasets of identical,
fixed resonators: predict the 2 * num_res positions [x, y] from the 40-120 Hz ERP.

Run as modules (the folder name starts with a digit, so it is imported by name
through ``python -m``):

    python -m erp_inverse.scripts.fixed_resonator.train_all
    python -m erp_inverse.scripts.fixed_resonator.evaluate
"""
