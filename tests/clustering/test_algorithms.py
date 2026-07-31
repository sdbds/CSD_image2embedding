import sys
import types
from unittest.mock import patch

import numpy as np
import torch

from csd_image2embedding.clustering import algorithms


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def test_kmeans_prefers_flash_backend_when_available():
    recorded = {}

    class FakeFlashKMeans:
        def __init__(self, d, k, seed=0, **kwargs):
            recorded.update(d=d, k=k, seed=seed, kwargs=kwargs)

        def fit(self, data):
            recorded["fit_data"] = _to_numpy(data)
            self.cluster_ids_b = np.array([[1, 0]], dtype=np.int64)
            self.centroids_b = np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32)
            return self

    with patch.dict(
        sys.modules,
        {"flash_kmeans": types.SimpleNamespace(FlashKMeans=FakeFlashKMeans)},
    ):
        with patch.object(
            algorithms,
            "KMeans",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("sklearn fallback should not run")
            ),
        ):
            result = algorithms.perform_kmeans(
                coords=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                k=2,
            )

    assert result.algorithm_name == "flash-kmeans"
    np.testing.assert_array_equal(result.labels_, [1, 0])
    np.testing.assert_allclose(result.cluster_centers_, [[0.0, 1.0], [1.0, 0.0]])


def test_kmeans_falls_back_only_for_documented_flash_runtime_failure():
    class FakeFlashKMeans:
        def __init__(self, **kwargs):
            del kwargs

        def fit(self, data):
            del data
            raise RuntimeError("flash failure")

    class FakeSklearnKMeans:
        def __init__(self, n_clusters, random_state):
            assert (n_clusters, random_state) == (2, 42)

        def fit(self, data):
            self.labels_ = np.array([0, 1], dtype=np.int32)
            self.cluster_centers_ = np.asarray(data, dtype=np.float32)
            return self

    with patch.dict(
        sys.modules,
        {"flash_kmeans": types.SimpleNamespace(FlashKMeans=FakeFlashKMeans)},
    ):
        with patch.object(algorithms, "KMeans", FakeSklearnKMeans):
            result = algorithms.perform_kmeans(coords=np.eye(2, dtype=np.float32), k=2)

    assert result.algorithm_name == "kmeans-sklearn-fallback"


def test_finch_passes_configuration_and_defaults_to_partition_one():
    recorded = {}

    def fake_finch(coords, req_clust=None, distance=None, verbose=None):
        recorded.update(
            coords=coords,
            req_clust=req_clust,
            distance=distance,
            verbose=verbose,
        )
        return np.array([[7, 3], [9, 3], [7, 4]], dtype=np.int32), None, None

    with patch.dict(sys.modules, {"finch": types.SimpleNamespace(FINCH=fake_finch)}):
        result = algorithms.perform_finch(
            coords=np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            req_clust=17,
        )

    assert recorded["req_clust"] == 17
    assert recorded["distance"] == "cosine"
    assert recorded["verbose"] is False
    np.testing.assert_array_equal(result.labels_, [0, 0, 1])


def test_finch_recovers_from_known_exact_cluster_library_bug():
    calls = []

    def fake_finch(coords, req_clust=None, distance=None, verbose=None):
        del coords, distance, verbose
        calls.append(req_clust)
        if req_clust is not None:
            raise UnboundLocalError("local variable 'requested_c' is not associated")
        return np.array([[10, 3], [20, 3], [30, 4], [10, 4]]), [3, 2], None

    with patch.dict(sys.modules, {"finch": types.SimpleNamespace(FINCH=fake_finch)}):
        result = algorithms.perform_finch(
            coords=np.eye(4, dtype=np.float32), req_clust=2
        )

    assert calls == [2, None]
    np.testing.assert_array_equal(result.labels_, [0, 0, 1, 1])


def test_default_k_does_not_force_finch_cluster_count():
    assert algorithms.resolve_finch_req_clust(40) is None
    assert algorithms.resolve_finch_req_clust(24) == 24
