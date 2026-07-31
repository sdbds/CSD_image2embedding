"""Embedding backend contracts and implementations."""

from __future__ import annotations

from .base import EmbeddingBackend, EmbeddingBatch, TextMode, validate_backend_mode

SUPPORTED_TEXT_MODES = {
    "csd": frozenset({"image-only"}),
    "siglip-dinov3": frozenset({"image-only", "caption-guided"}),
}


def _setting(settings, name: str, default=None):
    if isinstance(settings, dict):
        return settings.get(name, default)
    return getattr(settings, name, default)


def _create_csd_backend(settings):
    from .csd import CSDClipBackend

    return CSDClipBackend.from_pretrained(
        _setting(settings, "model_name", "yuxi-liu-wired/CSD"),
        _setting(settings, "processor_name", "openai/clip-vit-large-patch14"),
        device=_setting(settings, "device"),
        precision=_setting(settings, "precision", "auto"),
    )


def _create_siglip_dino_backend(settings):
    from .siglip_dinov3.backend import SiglipDinoBackend

    return SiglipDinoBackend.from_config(
        _setting(settings, "style_model_config"),
        checkpoint_override=_setting(settings, "style_model_checkpoint"),
        mode=_setting(settings, "text_mode", "image-only"),
        device=_setting(settings, "device"),
        precision=_setting(settings, "precision", "auto"),
    )


BACKEND_FACTORIES = {
    "csd": _create_csd_backend,
    "siglip-dinov3": _create_siglip_dino_backend,
}


def get_supported_text_modes(name: str) -> frozenset[str]:
    try:
        return SUPPORTED_TEXT_MODES[name]
    except KeyError as error:
        raise ValueError(f"Unknown embedding backend: {name}") from error


def create_backend(name: str, settings) -> EmbeddingBackend:
    try:
        factory = BACKEND_FACTORIES[name]
    except KeyError as error:
        raise ValueError(f"Unknown embedding backend: {name}") from error
    return factory(settings)


__all__ = [
    "BACKEND_FACTORIES",
    "EmbeddingBackend",
    "EmbeddingBatch",
    "TextMode",
    "create_backend",
    "get_supported_text_modes",
    "validate_backend_mode",
]
