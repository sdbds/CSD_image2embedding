import json

import numpy as np
import pandas as pd
import pytest

from csd_image2embedding.projection.manager import (
    DEFAULT_REDUCER,
    ProjectionManager,
    ProjectionSpec,
    load_projection_bundle,
)


def _base_dataframe():
    return pd.DataFrame(
        {
            "path": ["a.jpg", "b.jpg"],
            "style_embedding": [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ],
            "content_embedding": [
                np.array([0.5, 0.5], dtype=np.float32),
                np.array([0.3, 0.7], dtype=np.float32),
            ],
            "x1": [10.0, 20.0],
            "y1": [30.0, 40.0],
            "x2": [50.0, 60.0],
            "y2": [70.0, 80.0],
        }
    )


def _spec():
    return ProjectionSpec(
        name="tsne",
        parameters={
            "n_components": 2,
            "metric": "cosine",
            "init": "random",
            "learning_rate": "auto",
            "perplexity": 1,
        },
        random_state=42,
        implementation_version="test-version",
    )


def test_default_reducer_is_pacmap():
    assert DEFAULT_REDUCER == "pacmap"


def test_manager_persists_complete_metadata_and_reuses_coordinates(
    tmp_path, monkeypatch
):
    expected_style = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    expected_content = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    calls = []

    def fake_compute(style_embeddings, content_embeddings, spec):
        calls.append((style_embeddings.shape, content_embeddings.shape, spec))
        return {"style_xy": expected_style, "content_xy": expected_content}

    monkeypatch.setattr(
        "csd_image2embedding.projection.manager.compute_projection_bundle",
        fake_compute,
    )
    manager = ProjectionManager(
        _base_dataframe(),
        embedding_digest="embedding-manifest-a",
        cache_root=tmp_path,
    )

    projected = manager.get_projected_dataframe(_spec())

    assert len(calls) == 1
    np.testing.assert_allclose(projected[["x1", "y1"]], expected_style)
    cache_path = tmp_path / f"{_spec().digest('embedding-manifest-a')}.npz"
    loaded = load_projection_bundle(cache_path)
    assert loaded["metadata"] == _spec().metadata("embedding-manifest-a")

    second = ProjectionManager(
        _base_dataframe(),
        embedding_digest="embedding-manifest-a",
        cache_root=tmp_path,
    )
    second.get_projected_dataframe(_spec())
    assert len(calls) == 1


def test_manager_rejects_cache_with_incomplete_metadata(tmp_path):
    spec = _spec()
    digest = spec.digest("embedding-manifest-a")
    path = tmp_path / f"{digest}.npz"
    metadata = spec.metadata("embedding-manifest-a")
    metadata.pop("random_state")
    with path.open("wb") as stream:
        np.savez(
            stream,
            style_xy=np.zeros((2, 2), dtype=np.float32),
            content_xy=np.zeros((2, 2), dtype=np.float32),
            metadata_json=np.array(json.dumps(metadata)),
        )
    manager = ProjectionManager(
        _base_dataframe(),
        embedding_digest="embedding-manifest-a",
        cache_root=tmp_path,
    )

    with pytest.raises(ValueError, match="metadata"):
        manager.get_projected_dataframe(spec)


def test_legacy_spec_uses_stored_coordinates_without_raw_embeddings(tmp_path):
    base = _base_dataframe().drop(columns=["style_embedding", "content_embedding"])
    manager = ProjectionManager(
        base,
        embedding_digest="legacy-embedding",
        cache_root=tmp_path,
    )

    projected = manager.get_projected_dataframe(ProjectionSpec.legacy())

    pd.testing.assert_frame_equal(projected, base)
