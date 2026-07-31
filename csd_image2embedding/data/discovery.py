"""Deterministic image discovery with dependency-specific source digests."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
)
CaptionStatus = Literal["valid", "missing", "empty", "unreadable"]


@dataclass(frozen=True)
class SourceRecord:
    """One discovered image and its optional caption sidecar."""

    relative_path: str
    image_path: Path
    image_sha256: str
    caption: str | None
    caption_sha256: str | None
    caption_status: CaptionStatus


@dataclass(frozen=True)
class RejectedSourceRecord:
    """One image-like file rejected before it can enter a source snapshot."""

    relative_path: str
    reason: str


@dataclass(frozen=True)
class DirectorySnapshot:
    """A deterministic view of one directory source."""

    records: tuple[SourceRecord, ...]
    image_digest: str
    caption_digest: str
    caption_counts: dict[str, int]
    rejected_records: tuple[RejectedSourceRecord, ...] = ()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_json_rows(rows: list[list[str | None]]) -> str:
    payload = json.dumps(
        rows,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def read_caption(path: Path) -> tuple[str | None, CaptionStatus]:
    """Read and normalize a caption sidecar using supported encodings."""

    if not path.exists():
        return None, "missing"
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(encoding).strip()
            return (text, "valid") if text else (None, "empty")
        except UnicodeDecodeError:
            continue
    return None, "unreadable"


def discover_directory(root: Path) -> DirectorySnapshot:
    """Discover supported images and compute independent image/caption digests."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Image source is not a directory: {root}")

    image_paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )

    records: list[SourceRecord] = []
    rejected_records: list[RejectedSourceRecord] = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(root).as_posix()
        try:
            image_bytes = image_path.read_bytes()
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
        except (OSError, SyntaxError) as error:
            rejected_records.append(
                RejectedSourceRecord(
                    relative_path=relative_path,
                    reason=f"{type(error).__name__}: {error}",
                )
            )
            continue
        caption, caption_status = read_caption(image_path.with_suffix(".txt"))
        caption_sha256 = (
            _sha256_bytes(caption.encode("utf-8")) if caption is not None else None
        )
        records.append(
            SourceRecord(
                relative_path=relative_path,
                image_path=image_path,
                image_sha256=_sha256_bytes(image_bytes),
                caption=caption,
                caption_sha256=caption_sha256,
                caption_status=caption_status,
            )
        )

    image_rows = [[record.relative_path, record.image_sha256] for record in records]
    caption_rows = [
        [record.relative_path, record.caption_status, record.caption_sha256]
        for record in records
    ]
    caption_counts = {
        status: sum(record.caption_status == status for record in records)
        for status in ("valid", "missing", "empty", "unreadable")
    }
    if rejected_records:
        details = "\n".join(
            f"- {record.relative_path}: {record.reason}" for record in rejected_records
        )
        warnings.warn(
            f"Skipped unreadable images:\n{details}",
            UserWarning,
            stacklevel=2,
        )
    return DirectorySnapshot(
        records=tuple(records),
        image_digest=_digest_json_rows(image_rows),
        caption_digest=_digest_json_rows(caption_rows),
        caption_counts=caption_counts,
        rejected_records=tuple(rejected_records),
    )
