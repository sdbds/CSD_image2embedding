"""Coordinate selection and representative-image analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenericClusteringResult:
    labels_: np.ndarray
    algorithm_name: str
    cluster_centers_: np.ndarray | None = None
    noise_label: int | None = None
    implementation: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels_ = np.asarray(self.labels_, dtype=np.int32)
        if self.cluster_centers_ is not None:
            self.cluster_centers_ = np.asarray(self.cluster_centers_, dtype=np.float32)


def get_noise_label(clustering_result) -> int | None:
    return getattr(clustering_result, "noise_label", None)


def get_cluster_labels(clustering_result) -> list[int]:
    labels = np.asarray(clustering_result.labels_)
    noise_label = get_noise_label(clustering_result)
    return [
        int(label)
        for label in np.unique(labels)
        if noise_label is None or label != noise_label
    ]


def get_cluster_count(clustering_result) -> int:
    return len(get_cluster_labels(clustering_result))


def has_noise_cluster(clustering_result) -> bool:
    noise_label = get_noise_label(clustering_result)
    if noise_label is None:
        return False
    return bool(np.any(np.asarray(clustering_result.labels_) == noise_label))


def get_visual_column_names(feature_set: str = "1") -> tuple[str, str]:
    return ("x1", "y1") if feature_set == "1" else ("x2", "y2")


def get_embedding_column_name(feature_set: str = "1") -> str:
    return "style_embedding" if feature_set == "1" else "content_embedding"


def get_visual_coords(data, feature_set: str = "1") -> np.ndarray:
    x_column, y_column = get_visual_column_names(feature_set)
    return data[[x_column, y_column]].to_numpy(dtype=np.float32)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return np.asarray(vectors / safe_norms, dtype=np.float32)


def get_clustering_coords(data, feature_set: str = "1") -> np.ndarray:
    embedding_column = get_embedding_column_name(feature_set)
    if embedding_column in data.columns:
        vectors = np.vstack(
            [np.asarray(vector, dtype=np.float32) for vector in data[embedding_column]]
        )
        return _normalize_rows(vectors)
    return get_visual_coords(data, feature_set)


def _get_cluster_ids_and_centers(clustering_result, clustering_coords):
    labels = np.asarray(clustering_result.labels_)
    centers = getattr(clustering_result, "cluster_centers_", None)
    if centers is not None:
        centers = np.asarray(centers, dtype=np.float32)
        return np.arange(len(centers)), centers

    noise_label = get_noise_label(clustering_result)
    cluster_ids = [
        label
        for label in np.unique(labels)
        if noise_label is None or label != noise_label
    ]
    if not cluster_ids:
        return [], np.empty((0, clustering_coords.shape[1]), dtype=np.float32)
    centers = np.vstack(
        [clustering_coords[labels == label].mean(axis=0) for label in cluster_ids]
    ).astype(np.float32)
    return cluster_ids, centers


def summarize_clusters_for_display(
    data,
    clustering_result,
    feature_set: str = "1",
    clustering_coords=None,
    visual_coords=None,
) -> tuple[list[str], np.ndarray]:
    if clustering_coords is None:
        clustering_coords = get_clustering_coords(data, feature_set)
    if visual_coords is None:
        visual_coords = get_visual_coords(data, feature_set)
    clustering_coords = np.asarray(clustering_coords, dtype=np.float32)
    visual_coords = np.asarray(visual_coords, dtype=np.float32)
    labels = np.asarray(clustering_result.labels_)
    images = data["image"].tolist()

    cluster_ids, cluster_centers = _get_cluster_ids_and_centers(
        clustering_result, clustering_coords
    )
    representatives = []
    display_centers = []
    for cluster_id, cluster_center in zip(cluster_ids, cluster_centers, strict=True):
        cluster_indices = np.where(labels == cluster_id)[0]
        if not cluster_indices.size:
            continue
        display_centers.append(visual_coords[cluster_indices].mean(axis=0))
        distances = np.linalg.norm(
            clustering_coords[cluster_indices] - cluster_center, axis=1
        )
        representatives.append(images[cluster_indices[np.argmin(distances)]])

    if not display_centers:
        return [], np.empty((0, 2), dtype=np.float32)
    return representatives, np.vstack(display_centers).astype(np.float32)
