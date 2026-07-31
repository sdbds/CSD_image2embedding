"""Immutable generated-artifact builds published through an atomic pointer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

IDENTITY_KEYS = (
    "schema_version",
    "input_digest",
    "backend",
    "mode",
    "model_digest",
    "preprocessing_digest",
)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_synced_json(path: Path, value: Mapping[str, object]) -> bytes:
    payload = _canonical_json_bytes(value)
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


@dataclass(frozen=True)
class ArtifactIdentity:
    """Fields that determine whether an embedding artifact can be reused."""

    input_digest: str
    backend: str
    mode: str
    model_digest: str
    preprocessing_digest: str
    schema_version: int = 2

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_digest": self.input_digest,
            "backend": self.backend,
            "mode": self.mode,
            "model_digest": self.model_digest,
            "preprocessing_digest": self.preprocessing_digest,
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.as_dict())).hexdigest()


def validate_manifest(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
) -> None:
    """Fail with the identity fields that differ, ignoring creation metadata."""

    differing = sorted(
        key for key in IDENTITY_KEYS if expected.get(key) != actual.get(key)
    )
    if differing:
        raise ValueError(
            f"Artifact manifest identity differs for: {', '.join(differing)}"
        )


class ArtifactStore:
    """Store immutable builds and atomically select the current compatible build."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _logical_path(self, identity: ArtifactIdentity) -> Path:
        return (
            self.root
            / "embeddings"
            / f"v{identity.schema_version}"
            / identity.input_digest
            / identity.backend
            / identity.mode
            / identity.model_digest
        )

    @contextmanager
    def stage(self, identity: ArtifactIdentity) -> Iterator[Path]:
        builds = self._logical_path(identity) / "builds"
        builds.mkdir(parents=True, exist_ok=True)
        staged = builds / f".tmp-{uuid.uuid4().hex}"
        staged.mkdir()
        try:
            yield staged
        finally:
            if staged.exists():
                shutil.rmtree(staged)

    def publish(
        self,
        identity: ArtifactIdentity,
        staged_path: Path,
        manifest: Mapping[str, object],
    ) -> Path:
        """Complete an immutable build, then atomically switch its pointer."""

        staged_path = Path(staged_path)
        if not (staged_path / "data.lance").is_dir():
            raise ValueError("A staged artifact must contain a data.lance directory")

        identity_fields = identity.as_dict()
        conflicting = sorted(
            key
            for key, value in identity_fields.items()
            if key in manifest and manifest[key] != value
        )
        if conflicting:
            raise ValueError(
                "Manifest cannot override artifact identity fields: "
                + ", ".join(conflicting)
            )

        complete_manifest = {**manifest, **identity_fields}
        manifest_payload = _write_synced_json(
            staged_path / "manifest.json", complete_manifest
        )
        build_digest = hashlib.sha256(manifest_payload).hexdigest()

        logical_path = self._logical_path(identity)
        builds_path = logical_path / "builds"
        published = builds_path / build_digest
        if published.exists():
            existing = (published / "manifest.json").read_bytes()
            if existing != manifest_payload:
                raise ValueError(f"Artifact digest collision at {published}")
            shutil.rmtree(staged_path)
        else:
            staged_path.rename(published)

        pointer_tmp = logical_path / f".current-{uuid.uuid4().hex}.json"
        try:
            _write_synced_json(pointer_tmp, {"build_digest": build_digest})
            os.replace(pointer_tmp, logical_path / "current.json")
        finally:
            pointer_tmp.unlink(missing_ok=True)
        return published

    def resolve(self, identity: ArtifactIdentity) -> Path | None:
        """Resolve and validate the current build for an artifact identity."""

        logical_path = self._logical_path(identity)
        pointer_path = logical_path / "current.json"
        if not pointer_path.is_file():
            return None

        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        build_digest = pointer.get("build_digest")
        if not isinstance(build_digest, str) or len(build_digest) != 64:
            raise ValueError(f"Invalid artifact pointer: {pointer_path}")
        try:
            int(build_digest, 16)
        except ValueError as error:
            raise ValueError(f"Invalid artifact pointer: {pointer_path}") from error

        published = logical_path / "builds" / build_digest
        manifest_path = published / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"Artifact pointer references an incomplete build: {published}"
            )
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(identity.as_dict(), actual)
        return published
