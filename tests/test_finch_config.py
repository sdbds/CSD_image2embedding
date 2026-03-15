import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from dash_page import perform_finch, resolve_finch_req_clust


class FinchConfigTests(unittest.TestCase):
    def test_default_k_clusters_does_not_override_finch_req_clust(self):
        self.assertIsNone(resolve_finch_req_clust(40))

    def test_non_default_k_clusters_reuses_k_for_finch_req_clust(self):
        self.assertEqual(resolve_finch_req_clust(24), 24)

    def test_perform_finch_passes_req_clust(self):
        recorded = {}

        def fake_finch(coords, req_clust=None, distance=None, verbose=None):
            recorded["coords"] = coords
            recorded["req_clust"] = req_clust
            recorded["distance"] = distance
            recorded["verbose"] = verbose
            return np.array([[0], [1]], dtype=np.int32), None, None

        fake_module = types.SimpleNamespace(FINCH=fake_finch)

        with patch.dict(sys.modules, {"finch": fake_module}):
            result = perform_finch(
                coords=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                req_clust=17,
            )

        self.assertEqual(recorded["req_clust"], 17)
        self.assertEqual(recorded["distance"], "cosine")
        self.assertEqual(recorded["verbose"], False)
        np.testing.assert_array_equal(result.labels_, np.array([0, 1], dtype=np.int32))

    def test_perform_finch_defaults_to_partition_index_one(self):
        def fake_finch(coords, req_clust=None, distance=None, verbose=None):
            del coords, req_clust, distance, verbose
            return np.array([[7, 3], [9, 3], [7, 4]], dtype=np.int32), None, None

        fake_module = types.SimpleNamespace(FINCH=fake_finch)

        with patch.dict(sys.modules, {"finch": fake_module}):
            result = perform_finch(
                coords=np.array(
                    [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                    dtype=np.float32,
                )
            )

        np.testing.assert_array_equal(result.labels_, np.array([0, 0, 1], dtype=np.int32))

    def test_perform_finch_accepts_partition_index_override(self):
        def fake_finch(coords, req_clust=None, distance=None, verbose=None):
            del coords, req_clust, distance, verbose
            return np.array([[7, 3], [9, 3], [7, 4]], dtype=np.int32), None, None

        fake_module = types.SimpleNamespace(FINCH=fake_finch)

        with patch.dict(sys.modules, {"finch": fake_module}):
            result = perform_finch(
                coords=np.array(
                    [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                    dtype=np.float32,
                ),
                partition_index=0,
            )

        np.testing.assert_array_equal(result.labels_, np.array([0, 1, 0], dtype=np.int32))

    def test_perform_finch_recovers_from_exact_req_clust_bug_in_library(self):
        calls = []

        def fake_finch(coords, req_clust=None, distance=None, verbose=None):
            del coords, distance, verbose
            calls.append(req_clust)
            if req_clust is not None:
                raise UnboundLocalError(
                    "cannot access local variable 'requested_c' where it is not associated with a value"
                )
            return (
                np.array(
                    [
                        [10, 3],
                        [20, 3],
                        [30, 4],
                        [10, 4],
                    ],
                    dtype=np.int32,
                ),
                [3, 2],
                None,
            )

        fake_module = types.SimpleNamespace(FINCH=fake_finch)

        with patch.dict(sys.modules, {"finch": fake_module}):
            result = perform_finch(
                coords=np.array(
                    [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]],
                    dtype=np.float32,
                ),
                req_clust=2,
            )

        self.assertEqual(calls, [2, None])
        np.testing.assert_array_equal(result.labels_, np.array([0, 0, 1, 1], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
