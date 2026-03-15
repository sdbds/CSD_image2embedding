from pathlib import Path

import numpy as np

from projection_algorithms import (
    compute_projection_bundle,
    get_default_reducer,
    get_reducer_options,
)
from projection_cache import (
    build_dataset_fingerprint,
    build_projection_cache_path,
    load_projection_bundle,
    save_projection_bundle,
)


def _stack_vectors(series):
    return np.vstack([np.asarray(vector, dtype=np.float32) for vector in series])


class ProjectionManager:
    def __init__(self, base_df, cache_root=".projection_cache", random_state=42):
        self.base_df = base_df.copy()
        self.cache_root = Path(cache_root)
        self.random_state = random_state
        self._dataframes = {}

        self.has_raw_embeddings = {
            "style_embedding",
            "content_embedding",
        }.issubset(self.base_df.columns)

        if self.has_raw_embeddings:
            self.style_embeddings = _stack_vectors(self.base_df["style_embedding"])
            self.content_embeddings = _stack_vectors(self.base_df["content_embedding"])
            self.dataset_fingerprint = build_dataset_fingerprint(
                paths=self.base_df["path"].tolist(),
                style_embeddings=self.style_embeddings,
                content_embeddings=self.content_embeddings,
            )
        else:
            self.style_embeddings = None
            self.content_embeddings = None
            self.dataset_fingerprint = "legacy"

    def get_reducer_options(self):
        if not self.has_raw_embeddings:
            return [{"label": "Legacy Projection", "value": "legacy"}]
        return get_reducer_options()

    def get_default_reducer(self):
        if not self.has_raw_embeddings:
            return "legacy"
        return get_default_reducer()

    def get_projected_dataframe(self, reducer_name):
        if reducer_name in self._dataframes:
            return self._dataframes[reducer_name]

        if reducer_name == "legacy" or not self.has_raw_embeddings:
            projected_df = self.base_df.copy()
            self._dataframes[reducer_name] = projected_df
            return projected_df

        cache_path = build_projection_cache_path(
            cache_root=self.cache_root,
            dataset_fingerprint=self.dataset_fingerprint,
            reducer_name=reducer_name,
        )
        projection_bundle = load_projection_bundle(cache_path)
        if projection_bundle is None:
            projection_bundle = compute_projection_bundle(
                style_embeddings=self.style_embeddings,
                content_embeddings=self.content_embeddings,
                reducer_name=reducer_name,
                random_state=self.random_state,
            )
            save_projection_bundle(
                cache_root=self.cache_root,
                dataset_fingerprint=self.dataset_fingerprint,
                reducer_name=reducer_name,
                style_xy=projection_bundle["style_xy"],
                content_xy=projection_bundle["content_xy"],
                metadata={
                    "reducer": reducer_name,
                    "rows": len(self.base_df),
                    "style_dim": int(self.style_embeddings.shape[1]),
                    "content_dim": int(self.content_embeddings.shape[1]),
                },
            )

        projected_df = self.base_df.copy()
        projected_df["x1"] = projection_bundle["style_xy"][:, 0]
        projected_df["y1"] = projection_bundle["style_xy"][:, 1]
        projected_df["x2"] = projection_bundle["content_xy"][:, 0]
        projected_df["y2"] = projection_bundle["content_xy"][:, 1]
        self._dataframes[reducer_name] = projected_df
        return projected_df
