"""Peak-aware Set Transformer Operator (STO) for ERP spectrum prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from utils.neural_operator_utils import (
    MLP,
    physics_aware_resonator_features,
    resolve_activation,
    run_operator_experiment,
)


class DetuningCrossAttention(nn.Module):
    """Multi-head frequency-to-resonator attention with learned detuning bias."""

    def __init__(
        self,
        width: int,
        heads: int,
        dropout: float = 0.0,
        activation: str | type[nn.Module] = "silu",
    ) -> None:
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
        # raw [m,k,f_t,x,y], query f, delta, |delta| -> one bias per head
        self.bias_net = MLP([8, width // 2, heads], activation=resolve_activation(activation))
        self.attn_dropout = nn.Dropout(dropout)

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
        f_t = configuration[:, None, :, 2:3].expand(b, f, n, 1)
        delta = query_f - f_t
        bias_features = torch.cat((raw, query_f, delta, delta.abs()), dim=-1)
        bias = self.bias_net(bias_features).permute(0, 3, 1, 2)
        attention = torch.softmax(score + bias, dim=-1)
        attention = self.attn_dropout(attention)

        attended = torch.einsum("bhfn,bhnd->bhfd", attention, v)
        attended = attended.transpose(1, 2).contiguous().view(b, f, self.width)
        return self.out_proj(attended)


class FrequencyMixer(nn.Module):
    """Memory-efficient local/dilated frequency interaction after cross-attention."""

    def __init__(self, width: int, activation: str | type[nn.Module] = "gelu") -> None:
        super().__init__()
        self.local = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.dilated = nn.Conv1d(width, width, kernel_size=3, padding=2, dilation=2)
        self.norm = nn.GroupNorm(1, width)
        self.activation = resolve_activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x + self.local(x) + self.dilated(x)))


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
        dropout: float = 0.1,
        activation: str | type[nn.Module] = "gelu",
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.modal_harmonics = int(modal_harmonics)
        activation_cls = resolve_activation(activation)
        node_input_dim = 5 + 2 * self.modal_harmonics + self.modal_harmonics**2
        self.node_lift = MLP([node_input_dim, width, width], activation=activation_cls)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff_dim,
            # Was hard-coded to 0.0: with only num_res=3 resonator tokens,
            # self/cross-attention has more than enough capacity to overfit
            # once the easy gains are exhausted (visible as validation loss
            # rising again in the second half of training).
            dropout=dropout,
            # nn.TransformerEncoderLayer accepts a Callable[[Tensor],Tensor]
            # in addition to the "relu"/"gelu" strings, so an instantiated
            # activation module works for any choice, not just those two.
            activation=activation_cls(),
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.frequency_query = MLP([1, width, width], activation=activation_cls)
        self.cross_attention = DetuningCrossAttention(
            width, heads, dropout=dropout, activation=activation_cls
        )
        self.cross_norm = nn.LayerNorm(width)
        self.frequency_mixer = FrequencyMixer(width, activation=activation_cls)
        self.output = MLP([width, width, width // 2, 1], activation=activation_cls)

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


# width/ff_dim scaled down from the ~550K-matched config to the project's
# new ~110K budget (was width=132 -> ~554K), found via direct integer search
# rather than a single scale factor since width must stay divisible by
# heads=4 (kept unchanged at 4).
DEFAULT_MODEL_CONFIG = {
    "width": 56,
    "heads": 4,
    "depth": 2,
    "ff_dim": 144,
    "modal_harmonics": 4,
    "dropout": 0.1,
    "activation": "gelu",
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
