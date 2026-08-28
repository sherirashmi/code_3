"""Set Transformer Operator (STO) for ERP spectrum prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from utils.neural_operator_utils import MLP, run_operator_experiment


class SetTransformerOperator(nn.Module):
    """Attention-based permutation-invariant resonator-set operator.

    Resonator tokens interact through self-attention. Frequency embeddings are
    queries in a cross-attention layer over the resonator tokens, giving one ERP
    value at every requested frequency.
    """

    def __init__(
        self,
        num_res: int,
        width: int = 128,
        heads: int = 4,
        depth: int = 2,
        ff_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_res = int(num_res)
        self.node_lift = MLP([3, width, width], activation=nn.SiLU)
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
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(width)
        self.output = MLP([width, width, width // 2, 1], activation=nn.SiLU)

    def forward(self, configuration: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(self.node_lift(configuration))
        query = self.frequency_query(frequency)
        attended, _ = self.cross_attention(query, tokens, tokens, need_weights=False)
        return self.output(self.norm(query + attended))


def build_model(num_res: int, **kwargs) -> SetTransformerOperator:
    return SetTransformerOperator(num_res=num_res, **kwargs)


DEFAULT_MODEL_CONFIG = {
    "width": 128,
    "heads": 4,
    "depth": 2,
    "ff_dim": 256,
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
