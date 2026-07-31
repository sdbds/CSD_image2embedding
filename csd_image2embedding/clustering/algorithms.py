"""Clustering backends with narrow optional-dependency fallbacks."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .analysis import GenericClusteringResult, get_clustering_coords


def _to_numpy_array(value, dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _perform_sklearn_kmeans(coords, k: int, algorithm_name: str):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(coords)
    return GenericClusteringResult(
        labels_=model.labels_,
        cluster_centers_=model.cluster_centers_,
        algorithm_name=algorithm_name,
    )


def _perform_flash_kmeans(coords, k: int):
    import torch
    from flash_kmeans import FlashKMeans

    model = FlashKMeans(d=coords.shape[1], k=k, seed=42)
    model.fit(torch.as_tensor(coords, dtype=torch.float32))
    labels = _to_numpy_array(model.cluster_ids_b, np.int32)
    centers = _to_numpy_array(model.centroids_b, np.float32)
    if labels.ndim > 1:
        labels = labels[0]
    if centers.ndim > 2:
        centers = centers[0]
    return GenericClusteringResult(
        labels_=labels,
        cluster_centers_=centers,
        algorithm_name="flash-kmeans",
    )


def perform_kmeans(data=None, k: int = 40, feature_set: str = "1", coords=None):
    if coords is None:
        coords = get_clustering_coords(data, feature_set)
    coords = np.asarray(coords, dtype=np.float32)
    try:
        return _perform_flash_kmeans(coords, k)
    except (ImportError, RuntimeError):
        return _perform_sklearn_kmeans(coords, k, "kmeans-sklearn-fallback")


def perform_hdbscan(
    data=None,
    min_cluster_size: int = 5,
    feature_set: str = "1",
    coords=None,
):
    try:
        from hdbscan import HDBSCAN
    except ImportError as error:
        raise ImportError(
            "HDBSCAN clustering requires the 'hdbscan' package"
        ) from error
    if coords is None:
        coords = get_clustering_coords(data, feature_set)
    model = HDBSCAN(min_cluster_size=min_cluster_size).fit(coords)
    return GenericClusteringResult(
        labels_=model.labels_,
        algorithm_name="hdbscan",
        noise_label=-1,
    )


def resolve_finch_req_clust(
    k_clusters: int, default_k_clusters: int = 40
) -> int | None:
    return None if k_clusters == default_k_clusters else k_clusters


def _select_finch_labels(
    labels,
    num_clusters,
    requested_labels,
    requested_clusters,
    partition_index,
) -> np.ndarray:
    if requested_labels is not None:
        return np.asarray(requested_labels)
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return labels
    num_clusters = list(num_clusters or [])
    if requested_clusters is not None and num_clusters:
        if requested_clusters in num_clusters:
            return labels[:, num_clusters.index(requested_clusters)]
        if requested_clusters > num_clusters[0]:
            return labels[:, 0]
    if labels.shape[1] == 1:
        return labels[:, 0]
    if partition_index >= labels.shape[1]:
        return labels[:, -1]
    if partition_index < -labels.shape[1]:
        return labels[:, 0]
    return labels[:, partition_index]


def perform_finch(
    data=None,
    feature_set: str = "1",
    coords=None,
    req_clust=None,
    partition_index: int = 1,
):
    try:
        from finch import FINCH
    except ImportError as error:
        raise ImportError(
            "FINCH clustering requires the 'finch-clust' package"
        ) from error
    if coords is None:
        coords = get_clustering_coords(data, feature_set)
    try:
        labels, num_clusters, requested_labels = FINCH(
            coords,
            req_clust=req_clust,
            distance="cosine",
            verbose=False,
        )
    except UnboundLocalError as error:
        if req_clust is None or "requested_c" not in str(error):
            raise
        labels, num_clusters, requested_labels = FINCH(
            coords,
            req_clust=None,
            distance="cosine",
            verbose=False,
        )
    selected = _select_finch_labels(
        labels,
        num_clusters,
        requested_labels,
        req_clust,
        partition_index,
    )
    _, normalized = np.unique(selected, return_inverse=True)
    return GenericClusteringResult(normalized, "finch")
