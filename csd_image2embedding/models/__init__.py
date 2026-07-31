"""Embedding backend contracts and implementations."""

from .base import EmbeddingBackend, EmbeddingBatch, TextMode, validate_backend_mode

__all__ = [
    "EmbeddingBackend",
    "EmbeddingBatch",
    "TextMode",
    "validate_backend_mode",
]

