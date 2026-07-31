import subprocess
import sys
from dataclasses import replace

import pandas as pd
import pytest

from csd_image2embedding.clustering.analysis import GenericClusteringResult
from csd_image2embedding.clustering.export import (
    ExportIdentity,
    export_clustered_images,
)


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
