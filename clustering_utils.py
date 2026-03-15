import numpy as np


def get_visual_column_names(feature_set="1"):
    if feature_set == "1":
        return "x1", "y1"
    return "x2", "y2"


def get_embedding_column_name(feature_set="1"):
    if feature_set == "1":
        return "style_embedding"
    return "content_embedding"


def get_visual_coords(data, feature_set="1"):
    x_col, y_col = get_visual_column_names(feature_set)
    return data[[x_col, y_col]].to_numpy(dtype=np.float32)


def _normalize_rows(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return vectors / safe_norms


def get_clustering_coords(data, feature_set="1"):
    embedding_col = get_embedding_column_name(feature_set)
    if embedding_col in data.columns:
        vectors = np.vstack(
            [np.asarray(vector, dtype=np.float32) for vector in data[embedding_col]]
        )
        return _normalize_rows(vectors)
    return get_visual_coords(data, feature_set=feature_set)


def _get_cluster_ids_and_centers(clustering_result, clustering_coords):
    labels = np.asarray(clustering_result.labels_)
    centers_attr = getattr(clustering_result, "cluster_centers_", None)
    if centers_attr is not None:
        centers = np.asarray(centers_attr, dtype=np.float32)
        cluster_ids = np.arange(len(centers))
        return cluster_ids, centers

    cluster_ids = [label for label in np.unique(labels) if label != -1]
    if not cluster_ids:
        return [], np.empty((0, clustering_coords.shape[1]), dtype=np.float32)

    centers = np.vstack(
        [
            clustering_coords[labels == label].mean(axis=0)
            for label in cluster_ids
        ]
    ).astype(np.float32)
    return cluster_ids, centers


def summarize_clusters_for_display(
    data,
    clustering_result,
    feature_set="1",
    clustering_coords=None,
    visual_coords=None,
):
    if clustering_coords is None:
        clustering_coords = get_clustering_coords(data, feature_set=feature_set)
    if visual_coords is None:
        visual_coords = get_visual_coords(data, feature_set=feature_set)
    labels = np.asarray(clustering_result.labels_)
    images = data["image"].tolist()

    cluster_ids, cluster_centers = _get_cluster_ids_and_centers(
        clustering_result, clustering_coords
    )

    nearest_images = []
    display_centers = []

    for cluster_id, cluster_center in zip(cluster_ids, cluster_centers):
        cluster_indices = np.where(labels == cluster_id)[0]
        if cluster_indices.size == 0:
            continue

        display_centers.append(visual_coords[cluster_indices].mean(axis=0))
        distances = np.linalg.norm(
            clustering_coords[cluster_indices] - cluster_center,
            axis=1,
        )
        nearest_index = cluster_indices[np.argmin(distances)]
        nearest_images.append(images[nearest_index])

    if not display_centers:
        return [], np.empty((0, 2), dtype=np.float32)

    return nearest_images, np.vstack(display_centers).astype(np.float32)
