"""Conditional normalizing flow (RealNVP affine couplings) over the 2 * num_res positions.

Same model as ``erp_inverse.scripts.flow.ConditionalFlow``: an exactly
invertible map between the position design and a standard Gaussian latent,
conditioned on the ERP-band embedding; exact log p(positions | ERP).
"""

from erp_inverse.scripts.flow import ConditionalFlow

__all__ = ["ConditionalFlow"]
