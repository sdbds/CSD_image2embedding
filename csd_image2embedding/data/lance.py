"""Lance serialization for source images and generated embedding tables."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import uuid
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
SOURCE_SCHEMA_VERSION = 1


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
        try:
            with Image.open(io.BytesIO(binary_image)) as image:
                image.load()
                width, height = image.size
        except (OSError, SyntaxError):
            continue
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


def fingerprint_external_lance(dataset: lance.LanceDataset) -> str:
    """Fingerprint authoritative Lance rows without consulting source files."""

    schema_names = set(dataset.schema.names)
    path_column = next(
        (
            name
            for name in ("relative_path", "filename", "path")
            if name in schema_names
        ),
        None,
    )
    hash_column = next(
        (name for name in ("image_sha256", "hash") if name in schema_names),
        None,
    )
    if path_column is None or hash_column is None:
        raise ValueError(
            "External Lance input must contain a path column and a stored image hash"
        )

    table = dataset.to_table(columns=[path_column, hash_column])
    rows = [
        [path, image_hash]
        for path, image_hash in zip(
            table[path_column].to_pylist(),
            table[hash_column].to_pylist(),
            strict=True,
        )
    ]
    payload = {
        "schema": _canonical_schema(dataset.schema),
        "row_count": dataset.count_rows(),
        "path_column": path_column,
        "hash_column": hash_column,
        "rows": rows,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


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

    def __len__(self) -> int:
        return self.dataset.count_rows()

    def _row(self, index: int) -> dict[str, list[object]]:
        columns = ["filename", "image"]
        if "captions" in self._schema_names:
            columns.append("captions")
        return self.dataset.take([index], columns=columns).to_pydict()

    def __getitem__(self, index: int):
        row = self._row(index)
        path = str(row["filename"][0])
        image_bytes = row["image"][0]
        if image_bytes:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
        else:
            with Image.open(path) as source:
                image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        caption = str(row.get("captions", [""])[0]).strip() or None
        return path, image, caption


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
