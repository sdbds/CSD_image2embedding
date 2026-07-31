import numpy as np
import pytest

from csd_image2embedding.models.base import (
    EmbeddingBatch,
    precision_identity,
    validate_backend_mode,
)


def test_precision_identity_resolves_auto_instead_of_hashing_ambiguous_label():
    assert precision_identity("auto", None) == {
        "requested": "auto",
        "resolved": "fp32",
    }
    assert precision_identity("auto", "torch.float16") == {
        "requested": "auto",
        "resolved": "float16",
    }


def test_validate_backend_mode_rejects_caption_mode_for_csd():
    with pytest.raises(ValueError, match="csd.*caption-guided"):
        validate_backend_mode("csd", frozenset({"image-only"}), "caption-guided")


def test_embedding_batch_rejects_non_finite_values():
    batch = EmbeddingBatch(
        style_embeddings=np.array([[np.nan, 0.0]], dtype=np.float32),
        content_embeddings=np.ones((1, 2), dtype=np.float32),
        mode="image-only",
        backend="fake",
        model_fingerprint="model-1",
    )

    with pytest.raises(ValueError, match="finite"):
        batch.validate(expected_rows=1)


def test_embedding_batch_rejects_wrong_row_count():
    batch = EmbeddingBatch(
        style_embeddings=np.ones((1, 2), dtype=np.float32),
        content_embeddings=np.ones((1, 2), dtype=np.float32),
        mode="image-only",
        backend="fake",
        model_fingerprint="model-1",
    )

    with pytest.raises(ValueError, match="row count"):
        batch.validate(expected_rows=2)


def test_embedding_batch_rejects_zero_norm_rows():
    batch = EmbeddingBatch(
        style_embeddings=np.zeros((1, 2), dtype=np.float32),
        content_embeddings=np.ones((1, 2), dtype=np.float32),
        mode="image-only",
        backend="fake",
        model_fingerprint="model-1",
    )

    with pytest.raises(ValueError, match="nonzero norms"):
        batch.validate(expected_rows=1)
