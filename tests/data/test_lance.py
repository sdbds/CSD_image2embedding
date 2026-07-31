import hashlib
import io
import json

import lance
import pyarrow as pa
import pytest
from PIL import Image

from csd_image2embedding.data.discovery import discover_directory
from csd_image2embedding.data.lance import (
    SOURCE_MANIFEST_NAME,
    LanceImageDataset,
    fingerprint_external_lance,
    fingerprint_external_lance_inputs,
    write_source_snapshot,
)


def _write_png(path, color):
    Image.new("RGB", (2, 1), color).save(path, format="PNG")


def _png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 1), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_directory_snapshot_writes_source_identity_and_readable_rows(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    _write_png(source / "b.PNG", "blue")
    _write_png(source / "a.png", "red")
    (source / "a.txt").write_text("red square", encoding="utf-8")
    snapshot = discover_directory(source)

    output = tmp_path / "source.lance"
    dataset = write_source_snapshot(snapshot, output)

    manifest = json.loads((output / SOURCE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["image_digest"] == snapshot.image_digest
    assert manifest["caption_digest"] == snapshot.caption_digest
    assert manifest["row_count"] == 2
    assert dataset.to_table(columns=["relative_path"])["relative_path"].to_pylist() == [
        "a.png",
        "b.PNG",
    ]

    image_path, image, caption = LanceImageDataset(dataset)[0]
    assert image_path.endswith("a.png")
    assert image.size == (2, 1)
    assert caption == "red square"


def test_directory_snapshot_rejects_image_changed_after_discovery(tmp_path):
    source = tmp_path / "changing"
    source.mkdir()
    _write_png(source / "a.png", "red")
    snapshot = discover_directory(source)
    _write_png(source / "a.png", "blue")

    with pytest.raises(ValueError, match="changed during snapshot"):
        write_source_snapshot(snapshot, tmp_path / "changed.lance")


def test_external_lance_fingerprint_uses_stored_rows_not_source_files(tmp_path):
    source_image = tmp_path / "source.png"
    source_image.write_bytes(b"first")
    table = pa.table(
        {
            "filename": [str(source_image)],
            "hash": ["stored-hash"],
            "image": [b"stored-image"],
        }
    )
    dataset = lance.write_dataset(table, tmp_path / "external.lance")

    first = fingerprint_external_lance(dataset)
    source_image.write_bytes(b"changed-on-disk")
    second = fingerprint_external_lance(dataset)

    assert first == second


def test_external_lance_fingerprint_uses_the_embedded_image_bytes(tmp_path):
    first = lance.write_dataset(
        pa.table(
            {
                "filename": ["same.png"],
                "hash": ["same-declared-hash"],
                "image": [_png_bytes("red")],
            }
        ),
        tmp_path / "red.lance",
    )
    second = lance.write_dataset(
        pa.table(
            {
                "filename": ["same.png"],
                "hash": ["same-declared-hash"],
                "image": [_png_bytes("blue")],
            }
        ),
        tmp_path / "blue.lance",
    )

    assert fingerprint_external_lance(first) != fingerprint_external_lance(second)


def test_external_lance_fingerprints_image_bytes_in_streaming_batches(tmp_path):
    dataset = lance.write_dataset(
        pa.table(
            {
                "filename": ["a.png", "b.png"],
                "hash": ["a", "b"],
                "image": [_png_bytes("red"), _png_bytes("blue")],
            }
        ),
        tmp_path / "streamed.lance",
    )

    class StreamingDataset:
        schema = dataset.schema

        @staticmethod
        def count_rows():
            return dataset.count_rows()

        @staticmethod
        def to_batches(**kwargs):
            return dataset.to_batches(**kwargs)

        @staticmethod
        def to_table(columns):
            if "image" in columns:
                pytest.fail(
                    "embedded image bytes must not be materialized as one table"
                )
            return dataset.to_table(columns=columns)

    assert fingerprint_external_lance(StreamingDataset())


def test_external_lance_reader_accepts_a_path_without_embedded_bytes(tmp_path):
    image_path = tmp_path / "external.png"
    _write_png(image_path, "purple")
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    input_path = tmp_path / "external.lance"
    lance.write_dataset(
        pa.table(
            {
                "path": [str(image_path)],
                "image_sha256": [image_hash],
                "captions": [None],
            }
        ),
        input_path,
    )

    dataset = LanceImageDataset(input_path)
    path, image, caption = dataset[0]

    assert path == str(image_path)
    assert image.size == (2, 1)
    assert caption is None


def test_path_only_lance_rejects_a_file_changed_after_its_declared_hash(tmp_path):
    image_path = tmp_path / "external.png"
    _write_png(image_path, "red")
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    input_path = tmp_path / "external.lance"
    dataset = lance.write_dataset(
        pa.table(
            {
                "relative_path": ["stable/external.png"],
                "filename": [str(image_path)],
                "hash": [image_hash],
                "image": [b""],
            }
        ),
        input_path,
    )

    assert fingerprint_external_lance(dataset)
    _write_png(image_path, "blue")

    with pytest.raises(ValueError, match="declared image hash"):
        fingerprint_external_lance(dataset)
    with pytest.raises(ValueError, match="declared image hash"):
        LanceImageDataset(dataset)[0]


def test_lance_reader_uses_relative_record_id_but_reads_the_source_path(tmp_path):
    image_path = tmp_path / "external.png"
    _write_png(image_path, "purple")
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    dataset = lance.write_dataset(
        pa.table(
            {
                "relative_path": ["nested/external.png"],
                "filename": [str(image_path)],
                "hash": [image_hash],
                "image": [b""],
            }
        ),
        tmp_path / "relative.lance",
    )

    record_id, image, _ = LanceImageDataset(dataset)[0]

    assert record_id == "nested/external.png"
    assert image.size == (2, 1)


def test_external_lance_fingerprint_changes_with_record_id(tmp_path):
    first = lance.write_dataset(
        pa.table(
            {
                "filename": ["a.png"],
                "hash": ["declared"],
                "image": [_png_bytes("red")],
            }
        ),
        tmp_path / "first.lance",
    )
    second = lance.write_dataset(
        pa.table(
            {
                "filename": ["b.png"],
                "hash": ["declared"],
                "image": [_png_bytes("red")],
            }
        ),
        tmp_path / "second.lance",
    )

    assert fingerprint_external_lance(first) != fingerprint_external_lance(second)


def test_external_lance_keeps_image_and_caption_dependencies_separate(tmp_path):
    first = lance.write_dataset(
        pa.table(
            {
                "filename": ["a.png"],
                "hash": ["hash-a"],
                "image": [_png_bytes("red")],
                "captions": ["a lake"],
            }
        ),
        tmp_path / "first-caption.lance",
    )
    second = lance.write_dataset(
        pa.table(
            {
                "filename": ["a.png"],
                "hash": ["hash-a"],
                "image": [_png_bytes("red")],
                "captions": ["a mountain"],
            }
        ),
        tmp_path / "second-caption.lance",
    )

    first_identity = fingerprint_external_lance_inputs(first)
    second_identity = fingerprint_external_lance_inputs(second)

    assert first_identity.image_digest == second_identity.image_digest
    assert first_identity.caption_digest != second_identity.caption_digest
