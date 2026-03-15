import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cluster_result_utils import GenericClusteringResult
from clustering_utils import get_clustering_coords, summarize_clusters_for_display


class GetClusteringCoordsTests(unittest.TestCase):
    def test_prefers_high_dim_embedding_columns_when_available(self):
        data = pd.DataFrame(
            {
                "x1": [0.0, 100.0],
                "y1": [0.0, 100.0],
                "style_embedding": [
                    np.array([1.0, 0.0], dtype=np.float32),
                    np.array([0.0, 1.0], dtype=np.float32),
                ],
            }
        )

        coords = get_clustering_coords(data, feature_set="1")

        np.testing.assert_allclose(
            coords,
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

    def test_falls_back_to_visual_coordinates_when_embeddings_are_missing(self):
        data = pd.DataFrame(
            {
                "x2": [1.0, 2.0],
                "y2": [3.0, 4.0],
            }
        )

        coords = get_clustering_coords(data, feature_set="2")

        np.testing.assert_allclose(
            coords,
            np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        )


class SummarizeClustersForDisplayTests(unittest.TestCase):
    def test_uses_high_dim_centers_for_representatives_and_2d_centroids_for_display(self):
        data = pd.DataFrame(
            {
                "x1": [0.0, 2.0, 10.0, 12.0],
                "y1": [0.0, 0.0, 0.0, 0.0],
                "image": ["img-a", "img-b", "img-c", "img-d"],
                "style_embedding": [
                    np.array([1.0, 0.0], dtype=np.float32),
                    np.array([0.0, 1.0], dtype=np.float32),
                    np.array([-1.0, 0.0], dtype=np.float32),
                    np.array([0.0, -1.0], dtype=np.float32),
                ],
            }
        )
        model = SimpleNamespace(
            labels_=np.array([0, 0, 1, 1]),
            cluster_centers_=np.array([[0.9, 0.1], [-0.1, -0.9]], dtype=np.float32),
        )

        nearest_images, display_centers = summarize_clusters_for_display(
            data, model, feature_set="1"
        )

        self.assertEqual(nearest_images, ["img-a", "img-d"])
        np.testing.assert_allclose(
            display_centers,
            np.array([[1.0, 0.0], [11.0, 0.0]], dtype=np.float32),
        )

    def test_uses_supplied_clustering_and_visual_coords_without_dataframe_columns(self):
        data = pd.DataFrame({"image": ["img-a", "img-b"]})
        clustering_result = GenericClusteringResult(
            labels_=[0, 1],
            algorithm_name="generic",
        )

        nearest_images, display_centers = summarize_clusters_for_display(
            data,
            clustering_result,
            feature_set="1",
            clustering_coords=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            visual_coords=np.array([[10.0, 0.0], [20.0, 5.0]], dtype=np.float32),
        )

        self.assertEqual(nearest_images, ["img-a", "img-b"])
        np.testing.assert_allclose(
            display_centers,
            np.array([[10.0, 0.0], [20.0, 5.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
