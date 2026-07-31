"""Command line interface for creating an authoritative Lance snapshot."""

from __future__ import annotations

import argparse
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

from .discovery import discover_directory
from .lance import write_source_snapshot


def transform_directory(
    source: Path,
    output: Path,
    *,
    only_save_path: bool = False,
) -> Path:
    """Create a Lance snapshot without replacing an existing destination."""

    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}")

    snapshot = discover_directory(source)
    if not snapshot.records:
        raise ValueError(f"No supported images found in {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        write_source_snapshot(snapshot, staged, only_save_path=only_save_path)
        staged.rename(output)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic Lance snapshot from an image directory"
    )
    parser.add_argument("source", type=Path, help="Image directory to scan")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets.lance"),
        help="Destination Lance directory (default: datasets.lance)",
    )
    parser.add_argument(
        "--only-save-path",
        action="store_true",
        help="Store image paths instead of embedding image bytes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        transform_directory(
            args.source,
            args.output,
            only_save_path=args.only_save_path,
        )
    except ValueError as error:
        parser.error(str(error))
    return 0
