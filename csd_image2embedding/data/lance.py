"""Lance serialization for source images and generated embedding tables."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from PIL import Image

from .discovery import DirectorySnapshot

try:
    import pillow_avif  # noqa: F401
except ImportError:
    pass

try:
    import pillow_jxl  # noqa: F401
except ImportError:
    pass

try:
    from jxlpy import JXLImagePlugin  # noqa: F401
except ImportError:
    pass

SOURCE_MANIFEST_NAME = "_source_manifest.json"
SOURCE_SCHEMA_VERSION = 2

RECORD_ID_COLUMNS = ("relative_path", "filename", "path")
SOURCE_PATH_COLUMNS = ("filename", "path", "relative_path")
HASH_COLUMNS = ("image_sha256", "hash")


@dataclass(frozen=True)
class LanceInputIdentity:
    image_digest: str
    caption_digest: str
    caption_counts: dict[str, int]


@dataclass(frozen=True)
class LanceSourceRecord:
    """One authoritative Lance row resolved to the bytes consumers use."""

    record_id: str
    image_bytes: bytes
    image_sha256: str
    suffix: str
    source_path: Path | None
    caption: str | None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_source_manifest(output_path: Path, manifest: dict[str, object]) -> None:
    destination = output_path / SOURCE_MANIFEST_NAME
    temporary = output_path / f".{SOURCE_MANIFEST_NAME}-{uuid.uuid4().hex}"
    try:
        with temporary.open("wb") as stream:
            stream.write(_canonical_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_source_snapshot(
    snapshot: DirectorySnapshot,
    output_path: Path,
    *,
    only_save_path: bool = False,
) -> lance.LanceDataset:
    """Write a validated directory snapshot and its dependency digests."""

    output_path = Path(output_path)
    rows: list[dict[str, object]] = []
    for record in snapshot.records:
        binary_image = record.image_path.read_bytes()
        actual_hash = hashlib.sha256(binary_image).hexdigest()
        if actual_hash != record.image_sha256:
            raise ValueError(f"Image changed during snapshot: {record.image_path}")
        try:
            with Image.open(io.BytesIO(binary_image)) as image:
                image.load()
                width, height = image.size
        except (OSError, SyntaxError) as error:
            raise ValueError(
                f"Image became unreadable during snapshot: {record.image_path}"
            ) from error
        rows.append(
            {
                "filename": str(record.image_path),
                "relative_path": record.relative_path,
                "extension": record.image_path.suffix.lower(),
                "hash": record.image_sha256,
                "size": len(binary_image),
                "width": width,
                "height": height,
                "image": b"" if only_save_path else binary_image,
                "captions": record.caption or "",
                "caption_status": record.caption_status,
                "caption_hash": record.caption_sha256,
            }
        )

    if not rows:
        raise ValueError("The directory snapshot contains no readable images")

    table = pa.table(
        {
            "filename": pa.array([row["filename"] for row in rows], pa.string()),
            "relative_path": pa.array(
                [row["relative_path"] for row in rows], pa.string()
            ),
            "extension": pa.array([row["extension"] for row in rows], pa.string()),
            "hash": pa.array([row["hash"] for row in rows], pa.string()),
            "size": pa.array([row["size"] for row in rows], pa.int64()),
            "width": pa.array([row["width"] for row in rows], pa.int32()),
            "height": pa.array([row["height"] for row in rows], pa.int32()),
            "image": pa.array([row["image"] for row in rows], pa.binary()),
            "captions": pa.array([row["captions"] for row in rows], pa.string()),
            "caption_status": pa.array(
                [row["caption_status"] for row in rows], pa.string()
            ),
            "caption_hash": pa.array(
                [row["caption_hash"] for row in rows], pa.string()
            ),
        }
    )
    mode = "overwrite" if output_path.exists() else "create"
    dataset = lance.write_dataset(table, output_path, mode=mode)
    _write_source_manifest(
        output_path,
        {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "input_kind": "directory",
            "image_digest": snapshot.image_digest,
            "caption_digest": snapshot.caption_digest,
            "row_count": len(rows),
        },
    )
    return dataset


def _canonical_schema(schema: pa.Schema) -> dict[str, object]:
    metadata = {
        base64.b64encode(key).decode("ascii"): base64.b64encode(value).decode("ascii")
        for key, value in sorted((schema.metadata or {}).items())
    }
    return {
        "fields": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in schema
        ],
        "metadata": metadata,
    }


def _first_column(schema_names: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in schema_names), None)


def _unique_columns(*columns: str | None) -> list[str]:
    return list(dict.fromkeys(column for column in columns if column is not None))


def _read_path_bytes(
    path_value: object, declared_hash: object
) -> tuple[Path, bytes, str]:
    if path_value is None:
        raise ValueError("Path-only Lance row has no readable source path")
    path = Path(str(path_value)).expanduser()
    if not path.is_file():
        raise ValueError(f"Path-only Lance source does not exist: {path}")
    image_bytes = path.read_bytes()
    actual_hash = hashlib.sha256(image_bytes).hexdigest()
    if not isinstance(declared_hash, str) or actual_hash != declared_hash:
        raise ValueError(
            f"Path-only Lance source does not match its declared image hash: {path}"
        )
    return path.resolve(), image_bytes, actual_hash


def fingerprint_external_lance_inputs(
    dataset: lance.LanceDataset,
) -> LanceInputIdentity:
    """Fingerprint independent image and caption dependencies in Lance rows."""

    schema_names = set(dataset.schema.names)
    record_id_column = _first_column(schema_names, RECORD_ID_COLUMNS)
    source_path_column = _first_column(schema_names, SOURCE_PATH_COLUMNS)
    hash_column = _first_column(schema_names, HASH_COLUMNS)
    image_column = "image" if "image" in schema_names else None
    if record_id_column is None:
        raise ValueError("External Lance input must contain a supported record path")
    if image_column is None and (source_path_column is None or hash_column is None):
        raise ValueError(
            "External Lance input requires embedded image bytes or a path "
            "and image hash"
        )

    image_columns = _unique_columns(
        record_id_column, source_path_column, hash_column, image_column
    )
    row_count = dataset.count_rows()
    image_rows = []
    observed_rows = 0
    for batch in dataset.to_batches(
        columns=image_columns,
        batch_size_bytes=64 * 1024 * 1024,
        scan_in_order=True,
    ):
        image_batch = batch.to_pydict()
        for index in range(batch.num_rows):
            record_id = image_batch[record_id_column][index]
            embedded = (
                image_batch[image_column][index] if image_column is not None else None
            )
            if embedded:
                actual_hash = hashlib.sha256(embedded).hexdigest()
            else:
                source_value = (
                    image_batch[source_path_column][index]
                    if source_path_column is not None
                    else None
                )
                declared_hash = (
                    image_batch[hash_column][index] if hash_column is not None else None
                )
                _, _, actual_hash = _read_path_bytes(source_value, declared_hash)
            image_rows.append([record_id, actual_hash])
        observed_rows += batch.num_rows
    if observed_rows != row_count:
        raise ValueError("External Lance row count changed while fingerprinting")
    image_payload = {
        "schema": _canonical_schema(dataset.schema),
        "row_count": row_count,
        "record_id_column": record_id_column,
        "source_path_column": source_path_column,
        "image_column": image_column,
        "rows": image_rows,
    }
    image_digest = hashlib.sha256(_canonical_json_bytes(image_payload)).hexdigest()

    caption_column = "captions" if "captions" in schema_names else None
    status_column = "caption_status" if "caption_status" in schema_names else None
    caption_columns = [record_id_column]
    if caption_column is not None:
        caption_columns.append(caption_column)
    if status_column is not None:
        caption_columns.append(status_column)
    caption_table = dataset.to_table(columns=caption_columns)
    paths = caption_table[record_id_column].to_pylist()
    captions = (
        caption_table[caption_column].to_pylist()
        if caption_column is not None
        else [None] * len(paths)
    )
    recorded_statuses = (
        caption_table[status_column].to_pylist()
        if status_column is not None
        else [None] * len(paths)
    )
    counts = {"valid": 0, "missing": 0, "empty": 0, "unreadable": 0}
    caption_rows = []
    for path, caption, recorded_status in zip(
        paths, captions, recorded_statuses, strict=True
    ):
        if recorded_status == "unreadable":
            status = "unreadable"
            normalized = None
        elif caption is None:
            status = "empty" if recorded_status == "empty" else "missing"
            normalized = None
        elif not isinstance(caption, str):
            status = "unreadable"
            normalized = None
        elif caption.strip():
            status = "valid"
            normalized = caption.strip()
        else:
            status = "empty"
            normalized = None
        counts[status] += 1
        caption_hash = (
            hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if normalized is not None
            else None
        )
        caption_rows.append([path, status, caption_hash])
    caption_payload = {
        "row_count": dataset.count_rows(),
        "record_id_column": record_id_column,
        "caption_column": caption_column,
        "status_column": status_column,
        "rows": caption_rows,
    }
    caption_digest = hashlib.sha256(_canonical_json_bytes(caption_payload)).hexdigest()
    return LanceInputIdentity(image_digest, caption_digest, counts)


def fingerprint_external_lance(dataset: lance.LanceDataset) -> str:
    """Return the image-only identity of an authoritative Lance input."""

    return fingerprint_external_lance_inputs(dataset).image_digest


class LanceImageDataset:
    """Lazy Pillow view over an image-bearing Lance dataset."""

    def __init__(self, image_or_lance_path, transform=None):
        self.dataset = (
            image_or_lance_path
            if isinstance(image_or_lance_path, lance.LanceDataset)
            else lance.dataset(image_or_lance_path)
        )
        self.transform = transform
        self._schema_names = set(self.dataset.schema.names)
        self._record_id_column = _first_column(self._schema_names, RECORD_ID_COLUMNS)
        self._source_path_column = _first_column(
            self._schema_names, SOURCE_PATH_COLUMNS
        )
        self._hash_column = _first_column(self._schema_names, HASH_COLUMNS)
        if self._record_id_column is None:
            raise ValueError("Lance image input has no supported path column")
        self._image_column = "image" if "image" in self._schema_names else None

    def __len__(self) -> int:
        return self.dataset.count_rows()

    def _row(self, index: int) -> dict[str, list[object]]:
        columns = _unique_columns(
            self._record_id_column,
            self._source_path_column,
            self._hash_column,
            self._image_column,
            "captions" if "captions" in self._schema_names else None,
        )
        return self.dataset.take([index], columns=columns).to_pydict()

    def read_source(self, index: int) -> LanceSourceRecord:
        """Resolve one row to the exact bytes used by inference and export."""

        row = self._row(index)
        record_id = str(row[self._record_id_column][0])
        image_bytes = (
            row[self._image_column][0] if self._image_column is not None else None
        )
        source_value = (
            row[self._source_path_column][0]
            if self._source_path_column is not None
            else None
        )
        source_path = (
            Path(str(source_value)).expanduser() if source_value is not None else None
        )
        if image_bytes:
            image_bytes = bytes(image_bytes)
        else:
            declared_hash = (
                row[self._hash_column][0] if self._hash_column is not None else None
            )
            source_path, image_bytes, _ = _read_path_bytes(source_value, declared_hash)
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        suffix = Path(record_id).suffix
        if not suffix and source_path is not None:
            suffix = source_path.suffix
        caption_value = row.get("captions", [None])[0]
        caption = (
            caption_value.strip()
            if isinstance(caption_value, str) and caption_value.strip()
            else None
        )
        return LanceSourceRecord(
            record_id=record_id,
            image_bytes=image_bytes,
            image_sha256=image_sha256,
            suffix=suffix,
            source_path=source_path,
            caption=caption,
        )

    def resolve_verified_path(self, index: int, path_root: Path | None = None) -> Path:
        """Resolve a symlink target and verify it matches authoritative row bytes."""

        source = self.read_source(index)
        candidate = (
            Path(path_root) / Path(source.record_id)
            if path_root is not None
            else source.source_path
        )
        if candidate is None or not candidate.is_file():
            raise ValueError(
                f"No source file is available for Lance record: {source.record_id}"
            )
        candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if candidate_hash != source.image_sha256:
            raise ValueError(
                f"Symlink source does not match authoritative image bytes: {candidate}"
            )
        return candidate.resolve()

    def __getitem__(self, index: int):
        source_record = self.read_source(index)
        with Image.open(io.BytesIO(source_record.image_bytes)) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return source_record.record_id, image, source_record.caption


def _to_list_array(vectors) -> pa.Array:
    return pa.array(
        [np.asarray(vector, dtype=np.float32).tolist() for vector in vectors],
        type=pa.list_(pa.float32()),
    )


def build_embedding_table(
    paths,
    previews,
    style_embeddings,
    content_embeddings,
    style_projection,
    content_projection,
) -> pa.Table:
    """Build the stable table contract consumed by the dashboard."""

    return pa.table(
        {
            "path": pa.array(paths),
            "image": pa.array(previews),
            "style_embedding": _to_list_array(style_embeddings),
            "content_embedding": _to_list_array(content_embeddings),
            "x1": pa.array(style_projection[:, 0], type=pa.float32()),
            "y1": pa.array(style_projection[:, 1], type=pa.float32()),
            "x2": pa.array(content_projection[:, 0], type=pa.float32()),
            "y2": pa.array(content_projection[:, 1], type=pa.float32()),
        }
    )
