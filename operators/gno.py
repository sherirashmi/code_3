"""Graph Neural Operator (GNO) for ERP spectrum prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_operator_utils import MLP, run_operator_experiment


class GraphMessageLayer(nn.Module):
    """Complete-graph message passing over a small resonator set."""

    def __init__(self, width: int) -> None:
        super().__init__()
        # hi, hj and relative raw normalized [f_t,x,y].
        self.message = MLP([2 * width + 3, width, width], activation=nn.SiLU)
        self.update = MLP([2 * width, width, width], activation=nn.SiLU)
        self.norm = nn.LayerNorm(width)

    def forward(self, h: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        b, n, width = h.shape
        hi = h[:, :, None, :].expand(b, n, n, width)
        hj = h[:, None, :, :].expand(b, n, n, width)
        rel = features[:, None, :, :] - features[:, :, None, :]
        message = self.message(torch.cat((hi, hj, rel), dim=-1))

        # Remove self messages; for n=1 fall back to zero aggregate.
        if n > 1:
            mask = (~torch.eye(n, dtype=torch.bool, device=h.device))[None, :, :, None]
            aggregate = (message * mask).sum(dim=2) / float(n - 1)
        else:
            aggregate = torch.zeros_like(h)

        delta = self.update(torch.cat((h, aggregate), dim=-1))
        return F.silu(self.norm(h + delta))


class GNO(nn.Module):
    """Graph encoder plus frequency-query kernel aggregation.

    Resonators are graph nodes. Shared message passing is permutation equivariant;
    mean query aggregation makes the final ERP prediction permutation invariant.
    """

    def __init__(
        self,
        num_res: int,
        width: int = 128,
        depth: int = 3,
        frequency_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.node_lift = MLP([3, width, width], activation=nn.SiLU)
        self.layers = nn.ModuleList([GraphMessageLayer(width) for _ in range(depth)])
        self.frequency_encoder = MLP([1, frequency_dim, frequency_dim], activation=nn.SiLU)
        self.query_kernel = MLP(
            [width + 3 + frequency_dim, width, width, width],
            activation=nn.SiLU,
        )
        self.output = MLP([width, width // 2, 1], activation=nn.SiLU)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        h = self.node_lift(configuration)
        for layer in self.layers:
            h = layer(h, configuration)

        # Query every output frequency against every resonator node.
        freq = self.frequency_encoder(frequency)               # (B,F,Q)
        b, f, _ = freq.shape
        n = configuration.shape[1]
        node = h[:, None, :, :].expand(b, f, n, -1)
        raw = configuration[:, None, :, :].expand(b, f, n, -1)
        query = freq[:, :, None, :].expand(b, f, n, -1)
        kernel_values = self.query_kernel(torch.cat((node, raw, query), dim=-1))
        integral = kernel_values.mean(dim=2)
        return self.output(integral)


def build_model(num_res: int, **kwargs) -> GNO:
    return GNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 128,
    "depth": 3,
    "frequency_dim": 64,
}


def main(
    action: str = "train",
    num_configurations: int = 500,
    batch_size: int = 16,
    epochs: int = 200,
    learning_rate: float = 5e-4,
    dataset_file: str = "datasets/dataset_erp_ft.pth",
    regenerate_dataset: bool = False,
    seed: int = 727,
    configuration=None,
    plot: bool = True,
):
    return run_operator_experiment(
        operator_name="GNO",
        build_model=build_model,
        model_config=DEFAULT_MODEL_CONFIG,
        action=action,
        num_configurations=num_configurations,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        dataset_file=dataset_file,
        regenerate_dataset=regenerate_dataset,
        seed=seed,
        configuration=configuration,
        plot=plot,
    )


if __name__ == "__main__":
    results = main(action="train", num_configurations=500)
