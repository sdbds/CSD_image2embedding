import shutil
from dataclasses import replace
from pathlib import Path

import lance
import numpy as np
import pytest
from PIL import Image

from csd_image2embedding.artifacts import ArtifactStore
from csd_image2embedding.clustering.analysis import GenericClusteringResult
from csd_image2embedding.clustering.export import (
    ExportIdentity,
    export_clustered_images,
)
from csd_image2embedding.data.discovery import discover_directory
from csd_image2embedding.data.lance import LanceImageDataset, write_source_snapshot
from csd_image2embedding.models.base import EmbeddingBatch
from csd_image2embedding.projection.manager import ProjectionManager
from csd_image2embedding.workflow import WorkflowSettings, execute_workflow


class FakeBackend:
    name = "fake"
    fingerprint = "fake-model"
    preprocessing_fingerprint = "fake-preprocessing"
    supported_text_modes = frozenset({"image-only", "caption-guided"})

    def __init__(self, mode="image-only", *, fail=False):
        self.mode = mode
        self.fail = fail
        self.calls = 0

    def encode(self, images, captions=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("forced inference failure")
        rows = len(images)
        style = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (rows, 1))
        content = np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (rows, 1))
        return EmbeddingBatch(
            style_embeddings=style,
            content_embeddings=content,
            mode=self.mode,
            backend=self.name,
            model_fingerprint=self.fingerprint,
        )


def _write_image(path: Path, color: str):
    Image.new("RGB", (3, 2), color).save(path, format="PNG")


def _populate_source(source):
    source.mkdir()
    _write_image(source / "a.png", "red")
    _write_image(source / "b.png", "blue")
    (source / "a.txt").write_text("a lake", encoding="utf-8")
    (source / "b.txt").write_text("a city", encoding="utf-8")
    return source


def _source_fixture(tmp_path):
    return _populate_source(tmp_path / "images")


def _settings(source, artifact_root, mode="image-only", **changes):
    settings = WorkflowSettings(
        train_data_dir=source,
        dataset_path=None,
        artifact_root=artifact_root,
        backend="fake",
        text_mode=mode,
        batch_size=2,
        reducer="legacy",
        output_dir=None,
    )
    return replace(settings, **changes)


def _run_fixture(source, mode, artifact_root, external_lance=None, backend=None):
    settings = _settings(source, artifact_root, mode, dataset_path=external_lance)
    backend = backend or FakeBackend(mode)
    return execute_workflow(settings, backend=backend, launch_dashboard=False)


def test_replacing_image_changes_source_and_embedding_identities(tmp_path):
    source = _source_fixture(tmp_path)
    first = _run_fixture(source, "image-only", tmp_path / ".artifacts")
    _write_image(source / "a.png", "green")

    second = _run_fixture(source, "image-only", tmp_path / ".artifacts")

    assert first.source_identity != second.source_identity
    assert first.embedding_identity.digest() != second.embedding_identity.digest()


def test_caption_edit_does_not_invalidate_image_only_embedding(tmp_path):
    source = _source_fixture(tmp_path)
    backend = FakeBackend("image-only")
    first = _run_fixture(source, "image-only", tmp_path / ".artifacts", backend=backend)
    (source / "a.txt").write_text("a mountain", encoding="utf-8")

    second = _run_fixture(
        source, "image-only", tmp_path / ".artifacts", backend=backend
    )

    assert first.source_identity != second.source_identity
    assert first.embedding_identity == second.embedding_identity
    assert backend.calls == 1


def test_caption_edit_invalidates_caption_guided_embedding(tmp_path):
    source = _source_fixture(tmp_path)
    first = _run_fixture(source, "caption-guided", tmp_path / ".artifacts")
    (source / "a.txt").write_text("a mountain", encoding="utf-8")

    second = _run_fixture(source, "caption-guided", tmp_path / ".artifacts")

    assert first.embedding_identity.digest() != second.embedding_identity.digest()


def test_explicit_lance_is_authoritative_and_never_discovers_directory(
    tmp_path, monkeypatch
):
    source = _source_fixture(tmp_path)
    external = tmp_path / "external.lance"
    write_source_snapshot(discover_directory(source), external)
    monkeypatch.setattr(
        "csd_image2embedding.workflow.discover_directory",
        lambda path: pytest.fail(f"unexpected discovery: {path}"),
    )

    result = _run_fixture(
        source,
        "image-only",
        tmp_path / ".artifacts",
        external_lance=external,
    )

    assert result.input_kind == "lance"
    assert result.embedding_path.is_dir()


def test_equal_content_roots_export_from_snapshot_bytes_with_stable_record_ids(
    tmp_path,
):
    first_source = _populate_source(tmp_path / "first")
    second_source = _populate_source(tmp_path / "second")
    expected_bytes = {
        name: (second_source / name).read_bytes() for name in ("a.png", "b.png")
    }
    artifact_root = tmp_path / ".artifacts"

    first = _run_fixture(first_source, "image-only", artifact_root)
    shutil.rmtree(first_source)
    second = _run_fixture(second_source, "image-only", artifact_root)
    shutil.rmtree(second_source)

    assert first.source_path == second.source_path
    dataframe = (
        lance.dataset(second.embedding_path / "data.lance").to_table().to_pandas()
    )
    assert dataframe["path"].tolist() == ["a.png", "b.png"]

    output = export_clustered_images(
        dataframe,
        GenericClusteringResult([0, 1], "test-clusterer"),
        tmp_path / "export",
        ExportIdentity("emb", "proj", "test-clusterer", {}, 42),
        source_reader=LanceImageDataset(second.source_path),
    )

    assert (output / "class_0" / "image_0.png").read_bytes() == expected_bytes["a.png"]
    assert (output / "class_1" / "image_1.png").read_bytes() == expected_bytes["b.png"]


def test_default_directory_run_preserves_legacy_lance_directories(tmp_path):
    source = _source_fixture(tmp_path)
    legacy_source = tmp_path / "datasets.lance"
    legacy_embedding = tmp_path / "embeddings_csd.lance"
    legacy_source.mkdir()
    legacy_embedding.mkdir()
    (legacy_source / "sentinel.bin").write_bytes(b"source-before")
    (legacy_embedding / "sentinel.bin").write_bytes(b"embedding-before")

    result = _run_fixture(source, "image-only", tmp_path / ".artifacts")

    assert (legacy_source / "sentinel.bin").read_bytes() == b"source-before"
    assert (legacy_embedding / "sentinel.bin").read_bytes() == b"embedding-before"
    assert result.embedding_path.is_relative_to(tmp_path / ".artifacts")


def test_source_and_embedding_artifacts_use_the_post_audit_schema(tmp_path):
    source = _source_fixture(tmp_path)

    result = _run_fixture(source, "image-only", tmp_path / ".artifacts")

    assert "v2" in result.source_path.parts
    assert result.embedding_identity.schema_version == 3


def test_failed_rebuild_keeps_previous_current_artifact(tmp_path):
    source = _source_fixture(tmp_path)
    artifact_root = tmp_path / ".artifacts"
    first = _run_fixture(source, "image-only", artifact_root)
    store = ArtifactStore(artifact_root)
    current_before = store.resolve(first.embedding_identity)
    settings = _settings(source, artifact_root, rebuild=True)

    with pytest.raises(RuntimeError, match="forced inference failure"):
        execute_workflow(
            settings,
            backend=FakeBackend("image-only", fail=True),
            launch_dashboard=False,
        )

    assert store.resolve(first.embedding_identity) == current_before


def test_workflow_propagates_identities_through_projection_and_export(tmp_path):
    source = _source_fixture(tmp_path)

    result = _run_fixture(source, "image-only", tmp_path / ".artifacts")

    assert result.projection_identity == result.projection_spec.digest(
        result.embedding_manifest_digest
    )
    assert result.export_identity.embedding_digest == result.embedding_manifest_digest
    assert result.export_identity.projection_digest == result.projection_identity


def test_workflow_defers_projection_failures_to_the_requested_view(
    tmp_path, monkeypatch
):
    source = _source_fixture(tmp_path)
    monkeypatch.setattr(
        ProjectionManager,
        "get_projected_dataframe",
        lambda *args, **kwargs: pytest.fail(
            "projection ran before a view requested it"
        ),
    )

    result = _run_fixture(source, "image-only", tmp_path / ".artifacts")

    assert result.view_service.default_reducer == "legacy"
