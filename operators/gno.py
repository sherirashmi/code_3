"""Peak-aware Graph Neural Operator (GNO) for ERP spectrum prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_operator_utils import (
    MLP,
    physics_aware_resonator_features,
    run_operator_experiment,
)


class GraphMessageLayer(nn.Module):
    """Complete-graph message passing with geometric/frequency edge features."""

    def __init__(self, width: int) -> None:
        super().__init__()
        # hi, hj, relative [f_t,x,y], xy distance, |delta f_t|.
        self.message = MLP([2 * width + 5, width, width], activation=nn.SiLU)
        self.update = MLP([2 * width, width, width], activation=nn.SiLU)
        self.norm = nn.LayerNorm(width)

    def forward(self, h: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        b, n, width = h.shape
        hi = h[:, :, None, :].expand(b, n, n, width)
        hj = h[:, None, :, :].expand(b, n, n, width)
        rel = features[:, None, :, :] - features[:, :, None, :]
        distance = torch.linalg.vector_norm(rel[..., 1:3], dim=-1, keepdim=True)
        ft_gap = rel[..., 0:1].abs()
        edge = torch.cat((rel, distance, ft_gap), dim=-1)
        message = self.message(torch.cat((hi, hj, edge), dim=-1))

        if n > 1:
            mask = (~torch.eye(n, dtype=torch.bool, device=h.device))[None, :, :, None]
            aggregate = (message * mask).sum(dim=2) / float(n - 1)
        else:
            aggregate = torch.zeros_like(h)

        delta = self.update(torch.cat((h, aggregate), dim=-1))
        return F.silu(self.norm(h + delta))


class FrequencyRefinement1d(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1),
        )
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(x + self.net(x)))


class GNO(nn.Module):
    """Graph encoder with detuning-aware query kernels and learned node attention."""

    def __init__(
        self,
        num_res: int,
        width: int = 80,
        depth: int = 3,
        frequency_dim: int = 40,
        modal_harmonics: int = 4,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.modal_harmonics = int(modal_harmonics)
        node_input_dim = 3 + 2 * self.modal_harmonics + self.modal_harmonics**2
        self.node_lift = MLP([node_input_dim, width, width], activation=nn.SiLU)
        self.layers = nn.ModuleList([GraphMessageLayer(width) for _ in range(depth)])
        self.frequency_encoder = MLP(
            [1, frequency_dim, frequency_dim], activation=nn.SiLU
        )
        # node, raw resonator, frequency embedding, query f, delta, |delta|, delta^2
        query_input_dim = width + 3 + frequency_dim + 4
        self.query_kernel = MLP(
            [query_input_dim, width, width, width], activation=nn.SiLU
        )
        self.attention_score = MLP([width, width // 2, 1], activation=nn.SiLU)
        self.frequency_refinement = FrequencyRefinement1d(width)
        self.output = MLP([width, width // 2, 1], activation=nn.SiLU)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        node_features = physics_aware_resonator_features(
            configuration, harmonics=self.modal_harmonics
        )
        h = self.node_lift(node_features)
        for layer in self.layers:
            h = layer(h, configuration)

        freq = self.frequency_encoder(frequency)
        b, f, _ = freq.shape
        n = configuration.shape[1]
        node = h[:, None, :, :].expand(b, f, n, -1)
        raw = configuration[:, None, :, :].expand(b, f, n, -1)
        query_embedding = freq[:, :, None, :].expand(b, f, n, -1)
        query_frequency = frequency[:, :, None, :].expand(b, f, n, 1)
        f_t = configuration[:, None, :, 0:1].expand(b, f, n, 1)
        detuning = query_frequency - f_t

        pair = torch.cat(
            (
                node,
                raw,
                query_embedding,
                query_frequency,
                detuning,
                detuning.abs(),
                detuning.square(),
            ),
            dim=-1,
        )
        kernel_values = self.query_kernel(pair)
        weights = torch.softmax(self.attention_score(kernel_values), dim=2)
        integral = (weights * kernel_values).sum(dim=2)
        integral = self.frequency_refinement(integral.transpose(1, 2)).transpose(1, 2)
        return self.output(integral)


def build_model(num_res: int, **kwargs) -> GNO:
    return GNO(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 80,
    "depth": 3,
    "frequency_dim": 40,
    "modal_harmonics": 4,
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
