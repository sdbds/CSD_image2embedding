"""Frozen DINOv3 and SigLIP2 feature extractors."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, Dinov2Model, Dinov2Config


def _remap_dinov3_pth(raw: dict) -> dict:
    """Remap raw DINOv3 .pth state-dict keys to HuggingFace ViT format.

    HuggingFace has no Dinov3Model class; DINOv3 shares the same Transformer
    architecture as DINOv2, so we reuse Dinov2Model as the container and
    remap the key names from the official .pth checkpoint.
    """
    new: dict = {}

    def _put(dst, src_key):
        if src_key in raw:
            new[dst] = raw[src_key]

    _put("embeddings.cls_token",                          "cls_token")
    _put("embeddings.mask_token",                         "mask_token")
    _put("embeddings.patch_embeddings.projection.weight", "patch_embed.proj.weight")
    _put("embeddings.patch_embeddings.projection.bias",   "patch_embed.proj.bias")
    _put("layernorm.weight",                              "norm.weight")
    _put("layernorm.bias",                                "norm.bias")

    num_blocks = sum(1 for k in raw if k.startswith("blocks.") and k.endswith(".norm1.weight"))

    for i in range(num_blocks):
        prefix_src = f"blocks.{i}"
        prefix_dst = f"encoder.layer.{i}"

        for name in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            _put(f"{prefix_dst}.{name}", f"{prefix_src}.{name}")

        _put(f"{prefix_dst}.layer_scale1.lambda1", f"{prefix_src}.ls1.gamma")
        _put(f"{prefix_dst}.layer_scale2.lambda1", f"{prefix_src}.ls2.gamma")

        for name in ("mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"):
            _put(f"{prefix_dst}.{name}", f"{prefix_src}.{name}")

        _put(f"{prefix_dst}.attention.output.dense.weight", f"{prefix_src}.attn.proj.weight")
        _put(f"{prefix_dst}.attention.output.dense.bias",   f"{prefix_src}.attn.proj.bias")

        qkv_w_key = f"{prefix_src}.attn.qkv.weight"
        qkv_b_key = f"{prefix_src}.attn.qkv.bias"
        qkv_m_key = f"{prefix_src}.attn.qkv.bias_mask"
        if qkv_w_key in raw:
            q_w, k_w, v_w = raw[qkv_w_key].chunk(3, dim=0)
            new[f"{prefix_dst}.attention.attention.query.weight"] = q_w
            new[f"{prefix_dst}.attention.attention.key.weight"]   = k_w
            new[f"{prefix_dst}.attention.attention.value.weight"] = v_w
        if qkv_b_key in raw:
            bias = raw[qkv_b_key]
            if qkv_m_key in raw:
                bias = bias * raw[qkv_m_key]
            q_b, k_b, v_b = bias.chunk(3, dim=0)
            new[f"{prefix_dst}.attention.attention.query.bias"] = q_b
            new[f"{prefix_dst}.attention.attention.key.bias"]   = k_b
            new[f"{prefix_dst}.attention.attention.value.bias"] = v_b

    return new


def _build_dinov3_from_pth(pth_path: str) -> Dinov2Model:
    """Load DINOv3 weights from a local .pth file into a Dinov2Model container."""
    raw = torch.load(pth_path, map_location="cpu", weights_only=False)
    num_blocks = sum(1 for k in raw if k.startswith("blocks.") and k.endswith(".norm1.weight"))
    embed_dim  = raw["cls_token"].shape[-1]
    patch_size = raw["patch_embed.proj.weight"].shape[-1]
    mlp_dim    = raw["blocks.0.mlp.fc1.weight"].shape[0]

    config = Dinov2Config(
        hidden_size=embed_dim,
        num_attention_heads=embed_dim // 64,
        num_hidden_layers=num_blocks,
        patch_size=patch_size,
        image_size=224,
        intermediate_size=mlp_dim,
        layer_scale_init_value=1.0,
    )
    model = Dinov2Model(config)
    mapped = _remap_dinov3_pth(raw)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    non_pos_missing = [k for k in missing if "position_embeddings" not in k]
    if non_pos_missing:
        print(f"[DINOv3] Warning: unexpected missing keys: {non_pos_missing[:5]}")
    print(f"[DINOv3] Loaded from {pth_path}  (missing={len(missing)}, unexpected={len(unexpected)})")
    return model


class FrozenDINOv3(nn.Module):
    """Frozen DINOv3 ViT-L encoder. Extracts CLS token features."""

    def __init__(self, model_id: str):
        super().__init__()
        if model_id.endswith(".pth"):
            self.model = _build_dinov3_from_pth(model_id)
        else:
            self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True) -> "FrozenDINOv3":
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]


class FrozenSigLIP2(nn.Module):
    """Frozen SigLIP2 ViT-L encoder. Extracts image and text features."""

    def __init__(self, model_id: str):
        super().__init__()
        print(f"[SigLIP2] Loading model from {model_id} ...")
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"[SigLIP2] Loaded from {model_id}")

    def train(self, mode: bool = True) -> "FrozenSigLIP2":
        return super().train(False)

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeds = self.model.get_image_features(pixel_values=pixel_values)
        if not isinstance(image_embeds, torch.Tensor):
            image_embeds = image_embeds.pooler_output
        return nn.functional.normalize(image_embeds, dim=-1)

    @torch.no_grad()
    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_embeds = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        if not isinstance(text_embeds, torch.Tensor):
            text_embeds = text_embeds.pooler_output
        return nn.functional.normalize(text_embeds, dim=-1)
