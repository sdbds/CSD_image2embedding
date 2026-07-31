from types import SimpleNamespace

import numpy as np
import pandas as pd

from csd_image2embedding.clustering.analysis import (
    GenericClusteringResult,
    get_clustering_coords,
    summarize_clusters_for_display,
)


def test_clustering_prefers_normalized_embedding_columns():
    data = pd.DataFrame(
        {
            "x1": [0.0, 100.0],
            "y1": [0.0, 100.0],
            "style_embedding": [
                np.array([2.0, 0.0], dtype=np.float32),
                np.array([0.0, 4.0], dtype=np.float32),
            ],
        }
    )

    coords = get_clustering_coords(data, feature_set="1")

    np.testing.assert_allclose(coords, [[1.0, 0.0], [0.0, 1.0]])


def test_clustering_falls_back_to_visual_coordinates():
    data = pd.DataFrame({"x2": [1.0, 2.0], "y2": [3.0, 4.0]})

    coords = get_clustering_coords(data, feature_set="2")

    np.testing.assert_allclose(coords, [[1.0, 3.0], [2.0, 4.0]])


def test_representatives_use_embedding_centers_and_visual_centroids():
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
    result = SimpleNamespace(
        labels_=np.array([0, 0, 1, 1]),
        cluster_centers_=np.array([[0.9, 0.1], [-0.1, -0.9]], dtype=np.float32),
    )

    representatives, centers = summarize_clusters_for_display(
        data, result, feature_set="1"
    )

    assert representatives == ["img-a", "img-d"]
    np.testing.assert_allclose(centers, [[1.0, 0.0], [11.0, 0.0]])


def test_supplied_coordinate_arrays_do_not_require_dataframe_columns():
    data = pd.DataFrame({"image": ["img-a", "img-b"]})
    result = GenericClusteringResult([0, 1], "generic")

    representatives, centers = summarize_clusters_for_display(
        data,
        result,
        clustering_coords=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        visual_coords=np.array([[10.0, 0.0], [20.0, 5.0]], dtype=np.float32),
    )

    assert representatives == ["img-a", "img-b"]
    np.testing.assert_allclose(centers, [[10.0, 0.0], [20.0, 5.0]])
