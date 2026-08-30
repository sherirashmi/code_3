"""Peak-aware Set Transformer Operator (STO) for ERP spectrum prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.neural_operator_utils import (
    MLP,
    physics_aware_resonator_features,
    run_operator_experiment,
)


class DetuningCrossAttention(nn.Module):
    """Multi-head frequency-to-resonator attention with learned detuning bias."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("width must be divisible by heads.")
        self.width = int(width)
        self.heads = int(heads)
        self.head_dim = self.width // self.heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        # raw [f_t,x,y], query f, delta, |delta| -> one bias per head
        self.bias_net = MLP([6, width // 2, heads], activation=nn.SiLU)

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        configuration: torch.Tensor,
        frequency: torch.Tensor,
    ) -> torch.Tensor:
        b, f, _ = query.shape
        n = tokens.shape[1]

        q = self.q_proj(query).view(b, f, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(b, n, self.heads, self.head_dim).transpose(1, 2)

        score = torch.einsum("bhfd,bhnd->bhfn", q, k) / math.sqrt(float(self.head_dim))

        raw = configuration[:, None, :, :].expand(b, f, n, -1)
        query_f = frequency[:, :, None, :].expand(b, f, n, 1)
        f_t = configuration[:, None, :, 0:1].expand(b, f, n, 1)
        delta = query_f - f_t
        bias_features = torch.cat((raw, query_f, delta, delta.abs()), dim=-1)
        bias = self.bias_net(bias_features).permute(0, 3, 1, 2)
        attention = torch.softmax(score + bias, dim=-1)

        attended = torch.einsum("bhfn,bhnd->bhfd", attention, v)
        attended = attended.transpose(1, 2).contiguous().view(b, f, self.width)
        return self.out_proj(attended)


class FrequencyMixer(nn.Module):
    """Memory-efficient local/dilated frequency interaction after cross-attention."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.local = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.dilated = nn.Conv1d(width, width, kernel_size=3, padding=2, dilation=2)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(x + self.local(x) + self.dilated(x)))


class SetTransformerOperator(nn.Module):
    """Resonator self-attention + detuning-biased frequency cross-attention."""

    def __init__(
        self,
        num_res: int,
        width: int = 128,
        heads: int = 4,
        depth: int = 2,
        ff_dim: int = 256,
        modal_harmonics: int = 4,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.modal_harmonics = int(modal_harmonics)
        node_input_dim = 3 + 2 * self.modal_harmonics + self.modal_harmonics**2
        self.node_lift = MLP([node_input_dim, width, width], activation=nn.SiLU)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.frequency_query = MLP([1, width, width], activation=nn.SiLU)
        self.cross_attention = DetuningCrossAttention(width, heads)
        self.cross_norm = nn.LayerNorm(width)
        self.frequency_mixer = FrequencyMixer(width)
        self.output = MLP([width, width, width // 2, 1], activation=nn.SiLU)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        node_features = physics_aware_resonator_features(
            configuration, harmonics=self.modal_harmonics
        )
        tokens = self.encoder(self.node_lift(node_features))
        query = self.frequency_query(frequency)
        attended = self.cross_attention(query, tokens, configuration, frequency)
        h = self.cross_norm(query + attended)
        h = self.frequency_mixer(h.transpose(1, 2)).transpose(1, 2)
        return self.output(h)


def build_model(num_res: int, **kwargs) -> SetTransformerOperator:
    return SetTransformerOperator(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 128,
    "heads": 4,
    "depth": 2,
    "ff_dim": 256,
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
        operator_name="STO",
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
