"""StyleDecoupler: pure style extraction via orthogonal projection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as st_load

from sd_src.feature_extractors import FrozenDINOv3, FrozenSigLIP2
from sd_src.projector import AlignmentProjector


class StyleDecoupler(nn.Module):
    """Wraps frozen encoders + a loaded alignment projector (inference-only)."""

    def __init__(self, dino: FrozenDINOv3, siglip: FrozenSigLIP2, projector: AlignmentProjector):
        super().__init__()
        self.dino = dino
        self.siglip = siglip
        self.projector = projector
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True) -> "StyleDecoupler":
        return super().train(False)

    @torch.no_grad()
    def encode_image_dino(self, dino_pixels: torch.Tensor) -> torch.Tensor:
        """(B, D) aligned content features (c̄)."""
        return self.projector(self.dino(dino_pixels))

    @torch.no_grad()
    def encode_image_siglip(self, siglip_pixels: torch.Tensor) -> torch.Tensor:
        """(B, D) SigLIP2 image features (b̄)."""
        return self.siglip.get_image_features(siglip_pixels)

    @staticmethod
    def orthogonal_projection(s_r: torch.Tensor, c_r: torch.Tensor) -> torch.Tensor:
        """Confidence-weighted orthogonal projection (Eq. 5).

        alpha  = max(0, 1 - cosine_sim(s_r, c_r))
        s_pure = Norm(s_r - alpha * (s_r · c_r) * c_r)
        """
        sim = (s_r * c_r).sum(dim=-1, keepdim=True)
        alpha = torch.clamp(1.0 - sim, min=0.0)
        s_pure = s_r - alpha * (sim * c_r)
        return F.normalize(s_pure, dim=-1)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        dino_model_id: str,
        siglip_model_id: str,
        projector_hidden_dim: int = 2048,
        projector_num_layers: int = 3,
        projector_dropout: float = 0.0,
        dino_dim: int = 1024,
        siglip_dim: int = 1024,
        device: str | torch.device = "cpu",
    ) -> "StyleDecoupler":
        dino = FrozenDINOv3(dino_model_id)
        siglip = FrozenSigLIP2(siglip_model_id)
        projector = AlignmentProjector(
            input_dim=dino_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=siglip_dim,
            num_layers=projector_num_layers,
            dropout=projector_dropout,
        )
        tensors = st_load(checkpoint_path, device="cpu")
        proj_state = {k[len("projector."):]: v for k, v in tensors.items() if k.startswith("projector.")}
        projector.load_state_dict(proj_state)
        return cls(dino=dino, siglip=siglip, projector=projector).to(device)
