import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

import dash_page


def _to_numpy(data):
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return np.asarray(data)


class PerformKMeansTests(unittest.TestCase):
    def test_perform_kmeans_prefers_flash_kmeans_when_available(self):
        recorded = {}

        class FakeFlashKMeans:
            def __init__(self, d, k, seed=0, **kwargs):
                recorded["d"] = d
                recorded["k"] = k
                recorded["seed"] = seed
                recorded["kwargs"] = kwargs

            def fit(self, data):
                recorded["fit_data"] = _to_numpy(data)
                self.cluster_ids_b = np.array([[1, 0]], dtype=np.int64)
                self.centroids_b = np.array(
                    [[[0.0, 1.0], [1.0, 0.0]]],
                    dtype=np.float32,
                )
                return self

        fake_module = types.SimpleNamespace(FlashKMeans=FakeFlashKMeans)

        with patch.dict(sys.modules, {"flash_kmeans": fake_module}):
            with patch.object(
                dash_page,
                "KMeans",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("sklearn fallback should not be used")
                ),
            ):
                result = dash_page.perform_kmeans(
                    coords=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    k=2,
                )

        np.testing.assert_array_equal(result.labels_, np.array([1, 0], dtype=np.int32))
        np.testing.assert_allclose(
            result.cluster_centers_,
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(result.algorithm_name, "flash-kmeans")
        np.testing.assert_allclose(
            recorded["fit_data"],
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )

    def test_perform_kmeans_falls_back_to_sklearn_when_flash_kmeans_fit_fails(self):
        recorded = {"flash_fit_calls": 0, "sklearn_fit_calls": 0}

        class FakeFlashKMeans:
            def __init__(self, d, k, seed=0, **kwargs):
                recorded["flash_init"] = (d, k, seed, kwargs)

            def fit(self, data):
                del data
                recorded["flash_fit_calls"] += 1
                raise RuntimeError("flash failure")

        class FakeSklearnKMeans:
            def __init__(self, n_clusters, random_state):
                recorded["sklearn_init"] = (n_clusters, random_state)

            def fit(self, data):
                recorded["sklearn_fit_calls"] += 1
                recorded["sklearn_fit_data"] = np.asarray(data)
                self.labels_ = np.array([0, 1], dtype=np.int32)
                self.cluster_centers_ = np.array(
                    [[1.0, 0.0], [0.0, 1.0]],
                    dtype=np.float32,
                )
                return self

        fake_module = types.SimpleNamespace(FlashKMeans=FakeFlashKMeans)

        with patch.dict(sys.modules, {"flash_kmeans": fake_module}):
            with patch.object(dash_page, "KMeans", FakeSklearnKMeans):
                result = dash_page.perform_kmeans(
                    coords=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                    k=2,
                )

        self.assertEqual(recorded["flash_fit_calls"], 1)
        self.assertEqual(recorded["sklearn_fit_calls"], 1)
        self.assertEqual(result.algorithm_name, "kmeans-sklearn-fallback")
        np.testing.assert_array_equal(result.labels_, np.array([0, 1], dtype=np.int32))
        np.testing.assert_allclose(
            result.cluster_centers_,
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
