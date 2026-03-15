"""Dual-stream preprocessing pipeline for StyleDecoupler.

    Backbone   Resolution   Normalisation
    --------   ----------   -------------
    DINOv3     224×224      ImageNet mean/std
    SigLIP2    256×256      0.5 / 0.5
"""

from __future__ import annotations

import yaml
import torch
import numpy as np
from PIL import Image

from sd_src.transforms import (
    build_image_transform,
    IMAGENET_MEAN, IMAGENET_STD,
    SIGLIP2_MEAN,  SIGLIP2_STD,
)
from sd_model import load_style_decoupler, encode_image_only


class StyleDecouplerPipeline:
    """Wraps StyleDecoupler with dual-stream image preprocessing.

    Accepts a single PIL Image and returns CPU numpy arrays — matching
    the CSDCLIPPipeline output contract.
    """

    def __init__(self, model, dino_transform, siglip_transform, device: str = "cpu"):
        self.model = model
        self.dino_transform = dino_transform
        self.siglip_transform = siglip_transform
        self.device = device

    @classmethod
    def from_config(
        cls,
        config_path: str,
        checkpoint_override: str | None = None,
        device: str = "cpu",
    ) -> "StyleDecouplerPipeline":
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        dino_transform = build_image_transform(
            image_size=cfg["dino_image_size"],
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        siglip_transform = build_image_transform(
            image_size=cfg["siglip_image_size"],
            mean=SIGLIP2_MEAN,
            std=SIGLIP2_STD,
        )

        model, _ = load_style_decoupler(config_path, checkpoint_override, device)

        return cls(
            model=model,
            dino_transform=dino_transform,
            siglip_transform=siglip_transform,
            device=device,
        )

    def __call__(self, image: Image.Image) -> dict[str, np.ndarray]:
        """Run dual-stream inference on a single PIL image.

        Returns dict with "style_output", "content_output", "features" as (1, D) float32 arrays.
        """
        img_rgb = image.convert("RGB")
        dino_t   = self.dino_transform(img_rgb).unsqueeze(0).to(self.device)
        siglip_t = self.siglip_transform(img_rgb).unsqueeze(0).to(self.device)

        outputs = encode_image_only(self.model, dino_t, siglip_t)
        return {k: v.cpu().numpy() for k, v in outputs.items()}
