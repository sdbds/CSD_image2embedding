"""Identity-safe atomic export of clustered source images."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analysis import get_cluster_labels, get_noise_label, has_noise_cluster

RUN_MANIFEST_NAME = "run-manifest.json"
EXPORT_SCHEMA_VERSION = 2


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExportIdentity:
    embedding_digest: str
    projection_digest: str
    clusterer: str
    parameters: Mapping[str, object]
    random_state: int
    schema_version: int = EXPORT_SCHEMA_VERSION
    implementation: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "embedding_digest": self.embedding_digest,
            "projection_digest": self.projection_digest,
            "clusterer": self.clusterer,
            "implementation": dict(self.implementation),
            "parameters": dict(self.parameters),
            "random_state": self.random_state,
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.as_dict())).hexdigest()


def _write_manifest(
    path: Path,
    identity: ExportIdentity,
    files: list[dict[str, object]],
) -> None:
    payload = {
        "complete": True,
        "identity": identity.as_dict(),
        "identity_digest": identity.digest(),
        "files": files,
    }
    with path.open("wb") as stream:
        stream.write(_canonical_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())


def _validate_complete_export(output_root: Path, identity: ExportIdentity) -> None:
    manifest_path = output_root / RUN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            f"Export directory has no complete run manifest: {output_root}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != identity.as_dict():
        raise ValueError(
            f"Export directory has a different run identity: {output_root}"
        )
    files = manifest.get("files")
    if manifest.get("complete") is not True or not isinstance(files, list):
        raise ValueError(f"Export directory is incomplete: {output_root}")

    expected_paths = set()
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(
                f"Export manifest has an invalid file entry: {output_root}"
            )
        relative_path = entry["path"]
        expected_paths.add(relative_path)
        exported = output_root / relative_path
        expected_link = entry.get("mode") == "symlink"
        if expected_link != exported.is_symlink() or not exported.is_file():
            raise ValueError(f"Export directory is incomplete: {exported}")
        if _sha256_file(exported) != entry.get("sha256"):
            raise ValueError(f"Exported image digest mismatch: {exported}")

    actual_paths = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.name != RUN_MANIFEST_NAME and (path.is_file() or path.is_symlink())
    }
    if actual_paths != expected_paths:
        raise ValueError(f"Export directory file inventory mismatch: {output_root}")


def _destination_directory(staged: Path, label: int, noise_label: int | None) -> Path:
    name = (
        "noise"
        if noise_label is not None and label == noise_label
        else f"class_{label}"
    )
    destination = staged / name
    destination.mkdir(exist_ok=True)
    return destination


def _export_source(
    *,
    index: int,
    source_value: object,
    source_reader,
    source_root: Path | None,
    symlink: bool,
    destination_directory: Path,
) -> dict[str, object]:
    if source_reader is None:
        source_path = Path(str(source_value))
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Export source image does not exist: {source_path}"
            )
        source_path = source_path.resolve()
        image_bytes = source_path.read_bytes()
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        record_id = str(source_value)
        suffix = source_path.suffix
    else:
        source_record = source_reader.read_source(index)
        if source_record.record_id != str(source_value):
            raise ValueError(
                "Export source row does not match embedding record: "
                f"{source_record.record_id!r} != {source_value!r}"
            )
        image_bytes = source_record.image_bytes
        image_sha256 = source_record.image_sha256
        record_id = source_record.record_id
        suffix = source_record.suffix
        source_path = None

    suffix = suffix or ".bin"
    destination = destination_directory / f"image_{index}{suffix}"
    if symlink:
        if source_reader is not None:
            source_path = source_reader.resolve_verified_path(index, source_root)
        destination.symlink_to(source_path)
        mode = "symlink"
    else:
        destination.write_bytes(image_bytes)
        mode = "copy"
    return {
        "path": destination.relative_to(destination_directory.parent).as_posix(),
        "record_id": record_id,
        "sha256": image_sha256,
        "mode": mode,
    }


def export_clustered_images(
    data,
    clustering_result,
    output_root: Path,
    identity: ExportIdentity,
    symlink: bool = False,
    *,
    source_reader=None,
    source_root: Path | None = None,
) -> Path:
    """Publish a complete cluster export from authoritative source rows."""

    output_root = Path(output_root)
    labels = np.asarray(clustering_result.labels_)
    paths = data["path"].tolist()
    if len(labels) != len(paths):
        raise ValueError("Clustering labels do not match the exported image rows")
    if source_reader is not None and len(source_reader) != len(paths):
        raise ValueError("Export source rows do not match the embedding rows")

    if output_root.exists():
        _validate_complete_export(output_root, identity)
        return output_root

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staged = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    staged.mkdir()
    try:
        noise_label = get_noise_label(clustering_result)
        for label in get_cluster_labels(clustering_result):
            _destination_directory(staged, label, noise_label)
        if has_noise_cluster(clustering_result):
            _destination_directory(staged, noise_label, noise_label)

        files = []
        for index, (label, source_value) in enumerate(zip(labels, paths, strict=True)):
            destination_directory = _destination_directory(
                staged, int(label), noise_label
            )
            files.append(
                _export_source(
                    index=index,
                    source_value=source_value,
                    source_reader=source_reader,
                    source_root=source_root,
                    symlink=symlink,
                    destination_directory=destination_directory,
                )
            )
        _write_manifest(staged / RUN_MANIFEST_NAME, identity, files)
        try:
            staged.rename(output_root)
        except FileExistsError:
            _validate_complete_export(output_root, identity)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return output_root
