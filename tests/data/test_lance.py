import json

import lance
import pyarrow as pa
from PIL import Image

from csd_image2embedding.data.discovery import discover_directory
from csd_image2embedding.data.lance import (
    SOURCE_MANIFEST_NAME,
    LanceImageDataset,
    fingerprint_external_lance,
    write_source_snapshot,
)


def _write_png(path, color):
    Image.new("RGB", (2, 1), color).save(path, format="PNG")


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


def test_external_lance_fingerprint_changes_with_stored_path_or_hash(tmp_path):
    first = lance.write_dataset(
        pa.table({"filename": ["a.png"], "hash": ["hash-a"]}),
        tmp_path / "first.lance",
    )
    second = lance.write_dataset(
        pa.table({"filename": ["a.png"], "hash": ["hash-b"]}),
        tmp_path / "second.lance",
    )

    assert fingerprint_external_lance(first) != fingerprint_external_lance(second)
