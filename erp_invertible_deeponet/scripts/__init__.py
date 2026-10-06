"""Invertible DeepONet (Kaltenbach, Perdikaris & Koutsourelakis, Comput. Mech. 2023)
for the plate-resonator ERP: one network, forward (design -> ERP) and inverse
(ERP -> several designs) in the same weights.

    python -m erp_invertible_deeponet.scripts.train      # strict (Q = D) and padded (Q = 64) variants
    python -m erp_invertible_deeponet.scripts.evaluate
"""
