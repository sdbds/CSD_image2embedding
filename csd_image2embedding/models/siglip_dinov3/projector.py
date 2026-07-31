"""MLP projector aligning DINOv3 features to SigLIP2 space."""

import torch
import torch.nn as nn


class AlignmentProjector(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 2048,
        output_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        input_width = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(input_width, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_width = hidden_dim
        layers.append(nn.Linear(input_width, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.use_residual = input_dim == output_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.mlp(features)
        if self.use_residual:
            projected = projected + features
        projected = self.layer_norm(projected)
        return nn.functional.normalize(projected, dim=-1)
