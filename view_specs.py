def build_default_view_specs(model_label, k_clusters):
    titles = []
    params_list = []

    for suffix, feature_set in (("style", "1"), ("content", "2")):
        titles.extend(
            [
                f"[{model_label}] KMeans_{suffix}",
                f"[{model_label}] HDBSCAN_{suffix}",
                f"[{model_label}] FINCH_{suffix}",
            ]
        )
        params_list.extend(
            [
                {
                    "clusterer": "kmeans",
                    "k": k_clusters,
                    "feature_set": feature_set,
                },
                {"clusterer": "hdbscan", "feature_set": feature_set},
                {
                    "clusterer": "finch",
                    "k": k_clusters,
                    "feature_set": feature_set,
                },
            ]
        )

    return titles, params_list
