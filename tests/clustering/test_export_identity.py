import subprocess
import sys
from dataclasses import replace

import lance
import pandas as pd
import pyarrow as pa
import pytest
from PIL import Image

from csd_image2embedding.clustering.analysis import GenericClusteringResult
from csd_image2embedding.clustering.export import (
    ExportIdentity,
    export_clustered_images,
)
from csd_image2embedding.data.discovery import discover_directory
from csd_image2embedding.data.lance import LanceImageDataset, write_source_snapshot


def _fake_data(tmp_path):
    paths = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths.append(str(path))
    return pd.DataFrame({"path": paths})


def _fake_result():
    return GenericClusteringResult([5, 5, 7], "finch")


def test_importing_clustering_does_not_import_dash():
    code = (
        "import sys; import csd_image2embedding.clustering; "
        "assert 'dash' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_export_rejects_nonempty_directory_with_different_identity(tmp_path):
    first = ExportIdentity("emb-a", "proj-a", "kmeans", {"k": 2}, 42, 1)
    second = replace(first, parameters={"k": 3})
    output = tmp_path / "out"
    export_clustered_images(_fake_data(tmp_path), _fake_result(), output, first)

    with pytest.raises(ValueError, match="identity"):
        export_clustered_images(_fake_data(tmp_path), _fake_result(), output, second)


def test_matching_export_identity_is_idempotent_and_preserves_sparse_labels(tmp_path):
    identity = ExportIdentity("emb-a", "proj-a", "finch", {"partition": 1}, 42, 1)
    output = export_clustered_images(
        _fake_data(tmp_path), _fake_result(), tmp_path / "out", identity
    )
    export_clustered_images(_fake_data(tmp_path), _fake_result(), output, identity)

    assert (output / "run-manifest.json").is_file()
    assert (output / "class_5" / "image_0.jpg").is_file()
    assert (output / "class_7" / "image_2.jpg").is_file()
    assert not (output / "noise").exists()


def test_export_identity_includes_the_actual_clusterer_implementation():
    first = ExportIdentity(
        "emb",
        "proj",
        "kmeans",
        {"k": 2},
        42,
        implementation={"distribution": "scikit-learn", "version": "1.0"},
    )
    second = replace(
        first,
        implementation={"distribution": "scikit-learn", "version": "2.0"},
    )

    assert first.digest() != second.digest()


def test_failed_export_does_not_publish_a_partial_directory(tmp_path):
    data = _fake_data(tmp_path)
    (tmp_path / "b.jpg").unlink()
    output = tmp_path / "out"
    identity = ExportIdentity("emb", "proj", "finch", {}, 42)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        export_clustered_images(data, _fake_result(), output, identity)

    assert not output.exists()


def test_symlink_export_revalidates_the_active_source_root(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "a.png"
    Image.new("RGB", (2, 1), "red").save(image_path)
    snapshot_path = tmp_path / "source.lance"
    write_source_snapshot(discover_directory(source), snapshot_path)
    Image.new("RGB", (2, 1), "blue").save(image_path)
    output = tmp_path / "out"

    with pytest.raises(ValueError, match="authoritative image bytes"):
        export_clustered_images(
            pd.DataFrame({"path": ["a.png"]}),
            GenericClusteringResult([0], "test-clusterer"),
            output,
            ExportIdentity("emb", "proj", "test-clusterer", {}, 42),
            symlink=True,
            source_reader=LanceImageDataset(snapshot_path),
            source_root=source,
        )

    assert not output.exists()


def test_embedded_external_lance_resolves_a_verified_filename(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 1), "red").save(source)
    image_bytes = source.read_bytes()
    dataset_path = tmp_path / "external.lance"
    lance.write_dataset(
        pa.table(
            {
                "relative_path": ["source.png"],
                "filename": [str(source)],
                "hash": ["declarative-hash-is-not-authoritative"],
                "image": [image_bytes],
            }
        ),
        dataset_path,
    )

    reader = LanceImageDataset(dataset_path)

    assert reader.resolve_verified_path(0) == source.resolve()
