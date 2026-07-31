"""Shared contracts for image embedding backends."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

TextMode = Literal["image-only", "caption-guided"]


def precision_identity(requested: str, amp_dtype) -> dict[str, str]:
    """Describe the effective arithmetic precision used for cache identity."""

    resolved = "fp32" if amp_dtype is None else str(amp_dtype).removeprefix("torch.")
    return {"requested": requested, "resolved": resolved}


@dataclass(frozen=True)
class EmbeddingBatch:
    """Validated style and content embeddings produced by one backend call."""

    style_embeddings: np.ndarray
    content_embeddings: np.ndarray
    mode: TextMode
    backend: str
    model_fingerprint: str

    def validate(self, expected_rows: int) -> None:
        arrays = (self.style_embeddings, self.content_embeddings)
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("Embedding arrays must have rank 2")
        if any(len(array) != expected_rows for array in arrays):
            raise ValueError("Embedding row count does not match the input batch")
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("Embedding arrays must contain only finite values")
        if any(np.any(np.linalg.norm(array, axis=1) == 0) for array in arrays):
            raise ValueError("Embedding rows must have nonzero norms")


def validate_backend_mode(
    backend_name: str,
    supported_modes: frozenset[str],
    requested_mode: str,
) -> str:
    """Reject a requested text mode before loading heavyweight model assets."""

    if requested_mode not in supported_modes:
        raise ValueError(
            f"Backend '{backend_name}' does not support text mode '{requested_mode}'"
        )
    return requested_mode


class EmbeddingBackend(Protocol):
    """Runtime contract implemented by each embedding backend."""

    name: str
    fingerprint: str
    supported_text_modes: frozenset[str]

    def encode(self, images, captions=None) -> EmbeddingBatch:
        """Encode one batch of images and optional captions."""
        ...
