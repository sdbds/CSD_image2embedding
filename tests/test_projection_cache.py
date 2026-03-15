import tempfile
import unittest
from pathlib import Path

import numpy as np

from projection_cache import (
    DEFAULT_REDUCER,
    build_dataset_fingerprint,
    build_projection_cache_path,
    load_projection_bundle,
    save_projection_bundle,
)


class ProjectionCacheTests(unittest.TestCase):
    def test_default_reducer_is_pacmap(self):
        self.assertEqual(DEFAULT_REDUCER, "pacmap")

    def test_cache_round_trip_persists_style_and_content_coordinates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            fingerprint = "abc123"
            style_xy = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            content_xy = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

            cache_path = save_projection_bundle(
                cache_root=cache_root,
                dataset_fingerprint=fingerprint,
                reducer_name="pacmap",
                style_xy=style_xy,
                content_xy=content_xy,
                metadata={"reducer": "pacmap"},
            )

            loaded = load_projection_bundle(cache_path)

        self.assertTrue(cache_path.name.endswith("pacmap.npz"))
        np.testing.assert_allclose(loaded["style_xy"], style_xy)
        np.testing.assert_allclose(loaded["content_xy"], content_xy)
        self.assertEqual(loaded["metadata"]["reducer"], "pacmap")

    def test_fingerprint_changes_when_paths_change(self):
        style_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        content_embeddings = np.array([[0.5, 0.5], [0.3, 0.7]], dtype=np.float32)

        first = build_dataset_fingerprint(
            paths=["a.jpg", "b.jpg"],
            style_embeddings=style_embeddings,
            content_embeddings=content_embeddings,
        )
        second = build_dataset_fingerprint(
            paths=["a.jpg", "c.jpg"],
            style_embeddings=style_embeddings,
            content_embeddings=content_embeddings,
        )

        self.assertNotEqual(first, second)

    def test_cache_path_groups_by_dataset_fingerprint(self):
        path = build_projection_cache_path(
            cache_root=Path("E:/cache"),
            dataset_fingerprint="fingerprint-1",
            reducer_name="tsne",
        )

        self.assertEqual(
            path.as_posix(),
            "E:/cache/fingerprint-1/tsne.npz",
        )


if __name__ == "__main__":
    unittest.main()
