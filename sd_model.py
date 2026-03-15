"""StyleDecoupler image-only wrapper for CSD_image2embedding.

Image-only mode (no text descriptions):
    b_bar  = SigLIP2_image(Xi)              # 1024-dim, L2-norm (style+content)
    c_bar  = Projector(DINOv3(Xi))          # 1024-dim, L2-norm (content)
    s_pure = orthogonal_projection(b_bar, c_bar)  # 1024-dim, L2-norm (pure style)
"""

from __future__ import annotations

import os
import yaml
import torch

from sd_src.style_decoupler import StyleDecoupler


def _resolve(path: str, base: str) -> str:
    """Resolve *path* relative to *base* if it is not already absolute."""
    if not os.path.isabs(path):
        return os.path.normpath(os.path.join(base, path))
    return path


def load_style_decoupler(config_path: str, checkpoint_override: str | None = None, device: str = "cpu"):
    """Load StyleDecoupler from sd_config.yaml.

    Returns:
        (model, cfg) — StyleDecoupler instance and parsed config dict.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = cfg["model_base_path"]
    dino_model_id   = _resolve(cfg["dino_model_id"],   base)
    siglip_model_id = _resolve(cfg["siglip_model_id"], base)
    checkpoint_path = checkpoint_override or _resolve(cfg["checkpoint_path"], base)

    model = StyleDecoupler.from_checkpoint(
        checkpoint_path=checkpoint_path,
        dino_model_id=dino_model_id,
        siglip_model_id=siglip_model_id,
        projector_hidden_dim=cfg["projector_hidden_dim"],
        projector_num_layers=cfg["projector_num_layers"],
        dino_dim=cfg["dino_dim"],
        siglip_dim=cfg["siglip_dim"],
        device=device,
    )
    model.eval()
    return model, cfg


def encode_image_only(
    model: StyleDecoupler,
    dino_pixels: torch.Tensor,
    siglip_pixels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Image-only forward: s_r = b_bar, c_r = c_bar, then orthogonal projection."""
    with torch.no_grad():
        b_bar  = model.encode_image_siglip(siglip_pixels)
        c_bar  = model.encode_image_dino(dino_pixels)
        s_pure = model.orthogonal_projection(b_bar, c_bar)
    return {
        "style_output":   s_pure,
        "content_output": c_bar,
        "features":       b_bar,
    }
