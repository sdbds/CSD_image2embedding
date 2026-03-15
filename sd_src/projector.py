"""Lightweight MLP projector for aligning DINOv3 features to SigLIP2 space."""

import torch
import torch.nn as nn


class AlignmentProjector(nn.Module):
    """Configurable MLP projector with residual connection, GELU, and LayerNorm."""

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
        in_dim = input_dim
        for layer_idx in range(num_layers - 1):
            next_dim = hidden_dim
            layers.extend([
                nn.Linear(in_dim, next_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = next_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.use_residual = input_dim == output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        if self.use_residual:
            h = h + x
        h = self.layer_norm(h)
        return nn.functional.normalize(h, dim=-1)
