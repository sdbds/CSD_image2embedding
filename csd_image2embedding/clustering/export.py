"""Identity-safe export of clustered source images."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import get_cluster_labels, get_noise_label, has_noise_cluster

RUN_MANIFEST_NAME = "run-manifest.json"


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class ExportIdentity:
    embedding_digest: str
    projection_digest: str
    clusterer: str
    parameters: Mapping[str, object]
    random_state: int
    schema_version: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "embedding_digest": self.embedding_digest,
            "projection_digest": self.projection_digest,
            "clusterer": self.clusterer,
            "parameters": dict(self.parameters),
            "random_state": self.random_state,
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.as_dict())).hexdigest()


def _write_manifest(path: Path, identity: ExportIdentity) -> None:
    payload = {
        "identity": identity.as_dict(),
        "identity_digest": identity.digest(),
    }
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(_canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_or_create_manifest(output_root: Path, identity: ExportIdentity) -> None:
    manifest_path = output_root / RUN_MANIFEST_NAME
    existing_entries = list(output_root.iterdir())
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity.as_dict():
            raise ValueError(
                f"Export directory has a different run identity: {output_root}"
            )
        return
    if existing_entries:
        raise ValueError(
            f"Nonempty export directory has no run identity: {output_root}"
        )
    _write_manifest(manifest_path, identity)


def export_clustered_images(
    data,
    clustering_result,
    output_root: Path,
    identity: ExportIdentity,
    symlink: bool = False,
) -> Path:
    """Copy or link clustered images under a manifest-bound output directory."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _validate_or_create_manifest(output_root, identity)

    labels = np.asarray(clustering_result.labels_)
    paths = data["path"].tolist()
    if len(labels) != len(paths):
        raise ValueError("Clustering labels do not match the exported image rows")

    cluster_directories = {
        label: output_root / f"class_{label}"
        for label in get_cluster_labels(clustering_result)
    }
    for directory in cluster_directories.values():
        directory.mkdir(exist_ok=True)
    noise_label = get_noise_label(clustering_result)
    noise_directory = None
    if has_noise_cluster(clustering_result):
        noise_directory = output_root / "noise"
        noise_directory.mkdir(exist_ok=True)

    for index, (label, source_value) in enumerate(zip(labels, paths, strict=True)):
        source = Path(source_value)
        if not source.is_file():
            raise FileNotFoundError(f"Export source image does not exist: {source}")
        destination_directory = (
            noise_directory
            if noise_label is not None and label == noise_label
            else cluster_directories[int(label)]
        )
        destination = destination_directory / f"image_{index}.jpg"
        if destination.exists() or destination.is_symlink():
            continue
        if symlink:
            destination.symlink_to(source.resolve())
        else:
            shutil.copy2(source, destination)
    return output_root
