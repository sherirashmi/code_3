"""Mixture Density Network over the 2 * num_res positions.

Same model as ``erp_inverse_operators.mdn.MDN`` (spectrum encoder -> Gaussian
mixture over the flat design, exact log-density, closed-form sampling, k-means++
warm start of the component means), applied to the position-only design and the
40-120 Hz ERP band. Mixture components suit this problem: for a target ERP the
plausible position sets form a few compact clusters.
"""

from erp_inverse_operators.mdn import MDN

__all__ = ["MDN"]
