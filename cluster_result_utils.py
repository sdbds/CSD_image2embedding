from dataclasses import dataclass

import numpy as np


@dataclass
class GenericClusteringResult:
    labels_: np.ndarray
    algorithm_name: str
    cluster_centers_: np.ndarray | None = None
    noise_label: int | None = None

    def __post_init__(self):
        self.labels_ = np.asarray(self.labels_, dtype=np.int32)
        if self.cluster_centers_ is not None:
            self.cluster_centers_ = np.asarray(
                self.cluster_centers_,
                dtype=np.float32,
            )


def get_noise_label(clustering_result):
    return getattr(clustering_result, "noise_label", None)


def get_cluster_labels(clustering_result):
    labels = np.asarray(clustering_result.labels_)
    noise_label = get_noise_label(clustering_result)
    return [
        int(label)
        for label in np.unique(labels)
        if noise_label is None or label != noise_label
    ]


def get_cluster_count(clustering_result):
    return len(get_cluster_labels(clustering_result))


def has_noise_cluster(clustering_result):
    noise_label = get_noise_label(clustering_result)
    if noise_label is None:
        return False
    labels = np.asarray(clustering_result.labels_)
    return bool(np.any(labels == noise_label))
