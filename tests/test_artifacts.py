import json

import pytest

from csd_image2embedding.artifacts import (
    ArtifactIdentity,
    ArtifactStore,
    validate_manifest,
)


def fake_identity():
    return ArtifactIdentity(
        input_digest="input-a",
        backend="fake",
        mode="image-only",
        model_digest="model-a",
        preprocessing_digest="prep-a",
        schema_version=2,
    )


def test_publish_switches_pointer_only_after_complete_build(tmp_path):
    store = ArtifactStore(tmp_path)
    identity = fake_identity()
    with store.stage(identity) as staged:
        (staged / "data.lance").mkdir()
        (staged / "data.lance" / "data.bin").write_bytes(b"complete")
        published = store.publish(identity, staged, {"data_digest": "abc"})

    assert store.resolve(identity) == published
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_digest"] == "abc"
    assert manifest["backend"] == "fake"


def test_failed_stage_never_changes_current_pointer(tmp_path):
    store = ArtifactStore(tmp_path)
    identity = fake_identity()
    with store.stage(identity) as staged:
        (staged / "data.lance").mkdir()
        first = store.publish(identity, staged, {"data_digest": "first"})

    with pytest.raises(RuntimeError, match="stop"):
        with store.stage(identity) as staged:
            (staged / "partial").write_text("partial", encoding="utf-8")
            raise RuntimeError("stop")

    assert store.resolve(identity) == first
    assert not list(first.parent.glob(".tmp-*"))


def test_rebuild_preserves_previous_immutable_build(tmp_path):
    store = ArtifactStore(tmp_path)
    identity = fake_identity()
    with store.stage(identity) as staged:
        (staged / "data.lance").mkdir()
        first = store.publish(identity, staged, {"data_digest": "first"})
    with store.stage(identity) as staged:
        (staged / "data.lance").mkdir()
        second = store.publish(identity, staged, {"data_digest": "second"})

    assert first != second
    assert first.is_dir()
    assert store.resolve(identity) == second


def test_legacy_embedding_path_is_not_removed_or_reused(tmp_path):
    legacy = tmp_path / "embeddings_csd.lance"
    legacy.mkdir()
    store = ArtifactStore(tmp_path / ".artifacts")

    assert store.resolve(fake_identity()) is None
    assert legacy.exists()


def test_manifest_validation_reports_only_differing_identity_keys():
    expected = {
        **fake_identity().as_dict(),
        "created_at": "old",
        "command": "old-command",
    }
    actual = {
        **fake_identity().as_dict(),
        "backend": "other",
        "mode": "caption-guided",
        "created_at": "new",
        "command": "new-command",
    }

    with pytest.raises(ValueError, match=r"backend, mode$"):
        validate_manifest(expected, actual)


def test_manifest_validation_ignores_creation_metadata():
    expected = {
        **fake_identity().as_dict(),
        "created_at": "old",
        "command": "old-command",
    }
    actual = {
        **fake_identity().as_dict(),
        "created_at": "new",
        "command": "new-command",
    }

    validate_manifest(expected, actual)
