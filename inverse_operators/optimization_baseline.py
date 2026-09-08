"""Optimization-based inversion: no new network, reuse a trained forward operator.

Treats an already-trained forward operator (e.g. GNO, the best performer in
operators/) as a differentiable surrogate simulator and gradient-descends in
normalized design space to match a target spectrum, starting from many
random restarts. The spread of solutions found across restarts is a cheap,
approximate empirical posterior -- not a real density, and not as sharp or
theoretically grounded as the flow/diffusion/MDN models, but it costs no
training at all and is a fast way to sanity-check whether the forward model
is even accurate enough to invert before investing in a full generative
model. Recommended as the first thing to try.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import unflatten_configuration


@torch.enable_grad()
def invert_via_optimization(
    forward_model: nn.Module,
    frequency: torch.Tensor,
    target_spectrum: torch.Tensor,
    num_res: int,
    num_restarts: int = 32,
    steps: int = 300,
    lr: float = 0.05,
    init_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover a normalized design distribution for one target spectrum.

    Parameters
    ----------
    forward_model : trained operator (frozen), mapping (configuration, frequency) -> ERP.
    frequency : ``(n_freq, 1)`` normalized query frequencies (shared across restarts).
    target_spectrum : ``(n_freq, 1)`` normalized target ERP curve to match.
    num_restarts : number of independent random initializations (the "posterior sample size").

    Returns
    -------
    designs : ``(num_restarts, num_res, 5)`` normalized candidate configurations.
    final_losses : ``(num_restarts,)`` final per-restart MSE to the target (lower = better fit).
    """
    device = target_spectrum.device
    forward_model.eval()
    for p in forward_model.parameters():
        p.requires_grad_(False)

    flat_dim = num_res * 5
    x = (init_scale * torch.randn(num_restarts, flat_dim, device=device)).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    freq_batch = frequency[None, :, :].expand(num_restarts, -1, -1)
    target_batch = target_spectrum[None, :, :].expand(num_restarts, -1, -1)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        configuration = unflatten_configuration(x, num_res)
        prediction = forward_model(configuration, freq_batch)
        loss = ((prediction - target_batch) ** 2).mean(dim=(1, 2))
        loss.sum().backward()
        optimizer.step()

    with torch.no_grad():
        configuration = unflatten_configuration(x, num_res)
        prediction = forward_model(configuration, freq_batch)
        final_losses = ((prediction - target_batch) ** 2).mean(dim=(1, 2))

    return configuration.detach(), final_losses.detach()
