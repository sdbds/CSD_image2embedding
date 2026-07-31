"""Frozen DINOv3 and SigLIP2 feature extractors."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_DINOV3_HUB_REPO = (
    "facebookresearch/dinov3:6876159a11b4df116f30f667f8c9888617df0751"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_meta_repo(hub_repo: str) -> Path:
    """Cache official source without executing its broad hubconf module."""

    resolver = getattr(torch.hub, "_get_cache_or_reload", None)
    if resolver is None:
        raise RuntimeError(
            "This PyTorch version cannot cache GitHub Hub source without "
            "importing hubconf.py"
        )
    options = {
        "force_reload": False,
        "trust_repo": True,
        "verbose": True,
        "skip_validation": False,
    }
    if "calling_fn" in inspect.signature(resolver).parameters:
        options["calling_fn"] = "load"
    return Path(resolver(hub_repo, **options))


def _load_meta_dinov3(
    *,
    weights: str,
    hub_model: str,
    hub_repo: str,
) -> nn.Module:
    """Load raw weights through Meta's pinned official backbone factory."""

    repository = _resolve_meta_repo(hub_repo)
    repository_path = str(repository)
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    backbones = importlib.import_module("dinov3.hub.backbones")
    factory = getattr(backbones, hub_model, None)
    if factory is None or not callable(factory):
        raise ValueError(f"Unknown Meta DINOv3 Hub backbone: {hub_model}")
    model = factory(pretrained=False)
    state_dict = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


class FrozenDINOv3(nn.Module):
    """Frozen native DINOv3 encoder returning validated CLS features."""

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        *,
        hub_model: str = "dinov3_vitl16",
        hub_repo: str = DEFAULT_DINOV3_HUB_REPO,
        expected_dim: int = 1024,
    ):
        super().__init__()
        self.expected_dim = expected_dim
        model_path = Path(model_id).expanduser()
        if model_path.suffix.lower() == ".pth":
            if not model_path.is_file():
                raise FileNotFoundError(f"DINOv3 checkpoint not found: {model_path}")
            resolved_path = model_path.resolve()
            self.backend = "meta"
            self.model = _load_meta_dinov3(
                weights=str(resolved_path),
                hub_model=hub_model,
                hub_repo=hub_repo,
            )
            storage_tokens = getattr(self.model, "storage_tokens", None)
            expected_shape = (1, 4, expected_dim)
            if storage_tokens is None or tuple(storage_tokens.shape) != expected_shape:
                actual_shape = (
                    None if storage_tokens is None else tuple(storage_tokens.shape)
                )
                raise ValueError(
                    "Expected Meta DINOv3 with 4 register tokens shaped "
                    f"{expected_shape}, got {actual_shape}"
                )
            rope_periods = getattr(
                getattr(self.model, "rope_embed", None), "periods", None
            )
            if rope_periods is None:
                raise ValueError("Expected Meta DINOv3 RoPE periods")
            self.provenance = {
                "backend": self.backend,
                "model_ref": str(resolved_path),
                "hub_model": hub_model,
                "hub_repo": hub_repo,
                "checkpoint_sha256": _sha256_file(resolved_path),
                "architecture": {
                    "feature_dim": expected_dim,
                    "register_tokens": 4,
                    "positional_encoding": "rope",
                    "rope_period_count": int(rope_periods.numel()),
                },
            }
        else:
            self.backend = "huggingface"
            self.model = AutoModel.from_pretrained(model_id)
            model_type = getattr(
                getattr(self.model, "config", None), "model_type", None
            )
            if model_type != "dinov3_vit":
                raise ValueError(
                    "Expected a native DINOv3 Transformers model with "
                    f"config.model_type='dinov3_vit', got {model_type!r}"
                )
            hidden_size = getattr(self.model.config, "hidden_size", None)
            if hidden_size != expected_dim:
                raise ValueError(
                    "Expected DINOv3 feature dimension "
                    f"{expected_dim}, got {hidden_size}"
                )
            register_tokens = getattr(self.model.config, "num_register_tokens", None)
            if register_tokens != 4:
                raise ValueError(
                    "Expected native DINOv3 with 4 register tokens, "
                    f"got {register_tokens!r}"
                )
            self.provenance = {
                "backend": self.backend,
                "model_ref": model_id,
                "architecture": {
                    "feature_dim": expected_dim,
                    "register_tokens": 4,
                    "positional_encoding": "rope",
                },
            }
        self.model.eval()
        self.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> FrozenDINOv3:
        del mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.backend == "meta":
            outputs = self.model.forward_features(pixel_values)
            if not isinstance(outputs, dict) or "x_norm_clstoken" not in outputs:
                raise RuntimeError(
                    "Meta DINOv3 forward_features did not return x_norm_clstoken"
                )
            cls_features = outputs["x_norm_clstoken"]
        else:
            outputs = self.model(pixel_values=pixel_values)
            cls_features = outputs.last_hidden_state[:, 0]
        if cls_features.ndim != 2:
            raise RuntimeError(
                f"Expected DINOv3 CLS features with rank 2, got {cls_features.shape}"
            )
        if cls_features.shape[-1] != self.expected_dim:
            raise RuntimeError(
                f"Expected DINOv3 feature dimension {self.expected_dim}, "
                f"got {cls_features.shape[-1]}"
            )
        if not torch.isfinite(cls_features).all():
            raise RuntimeError("DINOv3 produced non-finite CLS features")
        return cls_features


class FrozenSigLIP2(nn.Module):
    """Frozen SigLIP2 encoder exposing image, text, and tokenizer APIs."""

    def __init__(self, model_id: str = "google/siglip2-large-patch16-256"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()
        self.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True) -> FrozenSigLIP2:
        del mode
        return super().train(False)

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeddings = self.model.get_image_features(pixel_values=pixel_values)
        if not isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings.pooler_output
        return nn.functional.normalize(image_embeddings, dim=-1)

    @torch.no_grad()
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_embeddings = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = text_embeddings.pooler_output
        return nn.functional.normalize(text_embeddings, dim=-1)
