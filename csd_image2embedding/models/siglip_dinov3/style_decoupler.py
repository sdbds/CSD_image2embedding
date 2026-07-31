"""Frozen dual encoder and alignment projector for style decoupling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from safetensors.torch import load_file as load_safetensors

from .feature_extractors import FrozenDINOv3, FrozenSigLIP2
from .projector import AlignmentProjector


class StyleDecoupler(nn.Module):
    def __init__(
        self,
        dino: FrozenDINOv3,
        siglip: FrozenSigLIP2,
        projector: AlignmentProjector,
    ):
        super().__init__()
        self.dino = dino
        self.siglip = siglip
        self.projector = projector
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> StyleDecoupler:
        del mode
        return super().train(False)

    @torch.no_grad()
    def encode_image_dino(self, dino_pixels: torch.Tensor) -> torch.Tensor:
        return self.projector(self.dino(dino_pixels))

    @torch.no_grad()
    def encode_image_siglip(self, siglip_pixels: torch.Tensor) -> torch.Tensor:
        return self.siglip.get_image_features(siglip_pixels)

    @torch.no_grad()
    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.siglip.get_text_features(input_ids, attention_mask)

    @staticmethod
    def combine_references(
        style_text: torch.Tensor,
        image: torch.Tensor,
        aligned_dino: torch.Tensor,
        content_text: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        style_reference = functional.normalize(style_text + image, dim=-1)
        content_reference = functional.normalize(content_text + aligned_dino, dim=-1)
        return style_reference, content_reference

    @staticmethod
    def orthogonal_projection(
        style_reference: torch.Tensor,
        content_reference: torch.Tensor,
    ) -> torch.Tensor:
        similarity = (style_reference * content_reference).sum(dim=-1, keepdim=True)
        alpha = torch.clamp(1.0 - similarity, min=0.0)
        projection = similarity * content_reference
        return functional.normalize(style_reference - alpha * projection, dim=-1)

    @torch.no_grad()
    def forward(
        self,
        dino_pixels: torch.Tensor,
        siglip_pixels: torch.Tensor,
        style_input_ids: torch.Tensor,
        style_attention_mask: torch.Tensor,
        content_input_ids: torch.Tensor,
        content_attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        image = self.encode_image_siglip(siglip_pixels)
        aligned_dino = self.encode_image_dino(dino_pixels)
        style_text = self.encode_text(style_input_ids, style_attention_mask)
        content_text = self.encode_text(content_input_ids, content_attention_mask)
        style_reference, content_reference = self.combine_references(
            style_text, image, aligned_dino, content_text
        )
        pure_style = self.orthogonal_projection(style_reference, content_reference)
        return {
            "s_pure": pure_style,
            "s_r": style_reference,
            "c_r": content_reference,
            "b_bar": image,
            "c_bar": aligned_dino,
            "a_bar": style_text,
            "d_bar": content_text,
        }

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
    ) -> StyleDecoupler:
        dino = FrozenDINOv3(dino_model_id, expected_dim=dino_dim)
        siglip = FrozenSigLIP2(siglip_model_id)
        projector = AlignmentProjector(
            input_dim=dino_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=siglip_dim,
            num_layers=projector_num_layers,
            dropout=projector_dropout,
        )
        tensors = load_safetensors(checkpoint_path, device="cpu")
        projector_state = {
            key.removeprefix("projector."): value
            for key, value in tensors.items()
            if key.startswith("projector.")
        }
        projector.load_state_dict(projector_state, strict=True)
        return cls(dino=dino, siglip=siglip, projector=projector).to(device)
