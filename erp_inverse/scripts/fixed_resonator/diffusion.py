"""Conditional DDPM (cosine noise schedule) over the 2 * num_res positions.

Same model as ``erp_inverse.scripts.diffusion.ConditionalDiffusion``: an MLP
denoiser conditioned on the ERP-band embedding and the timestep; samples by
running the reverse chain from noise. No tractable density, so it has no
target-blind "own pick" (its first sample is reported instead).
"""

from erp_inverse.scripts.diffusion import ConditionalDiffusion

__all__ = ["ConditionalDiffusion"]
