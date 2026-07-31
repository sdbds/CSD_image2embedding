"""Complete projection identities, persistence, and dataframe adaptation."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .algorithms import (
    DEFAULT_REDUCER,
    compute_projection_bundle,
    default_parameters,
    get_default_reducer,
    get_implementation_version,
    get_reducer_options,
)

PROJECTION_SCHEMA_VERSION = 2


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    parameters: Mapping[str, object]
    random_state: int
    implementation_version: str

    @classmethod
    def for_reducer(
        cls,
        name: str,
        *,
        num_samples: int,
        random_state: int = 42,
        parameters: Mapping[str, object] | None = None,
    ) -> ProjectionSpec:
        return cls(
            name=name,
            parameters=(
                dict(parameters)
                if parameters is not None
                else default_parameters(name, num_samples)
            ),
            random_state=random_state,
            implementation_version=get_implementation_version(name),
        )

    @classmethod
    def legacy(cls) -> ProjectionSpec:
        return cls("legacy", {}, 0, "builtin")

    def metadata(self, embedding_digest: str) -> dict[str, object]:
        return {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "embedding_digest": embedding_digest,
            "reducer": self.name,
            "parameters": dict(self.parameters),
            "random_state": self.random_state,
            "implementation_version": self.implementation_version,
        }

    def digest(self, embedding_digest: str) -> str:
        payload = _canonical_json(self.metadata(embedding_digest)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def save_projection_bundle(
    cache_path: Path,
    *,
    style_xy: np.ndarray,
    content_xy: np.ndarray,
    metadata: Mapping[str, object],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                style_xy=np.asarray(style_xy, dtype=np.float32),
                content_xy=np.asarray(content_xy, dtype=np.float32),
                metadata_json=np.array(_canonical_json(metadata)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)


def load_projection_bundle(cache_path: Path) -> dict[str, object] | None:
    if not cache_path.is_file():
        return None
    with np.load(cache_path, allow_pickle=False) as bundle:
        required = {"style_xy", "content_xy", "metadata_json"}
        if required - set(bundle.files):
            raise ValueError(f"Projection cache is incomplete: {cache_path}")
        return {
            "style_xy": bundle["style_xy"].astype(np.float32),
            "content_xy": bundle["content_xy"].astype(np.float32),
            "metadata": json.loads(str(bundle["metadata_json"].item())),
        }


def _stack_vectors(series) -> np.ndarray:
    return np.vstack([np.asarray(vector, dtype=np.float32) for vector in series])


class ProjectionManager:
    def __init__(
        self,
        base_df,
        *,
        embedding_digest: str,
        cache_root: Path = Path(".artifacts/projections/v2"),
    ):
        if not embedding_digest:
            raise ValueError("Projection caching requires an embedding manifest digest")
        self.base_df = base_df.copy()
        self.embedding_digest = embedding_digest
        self.cache_root = Path(cache_root)
        self._dataframes: dict[str, object] = {}
        self.has_raw_embeddings = {
            "style_embedding",
            "content_embedding",
        }.issubset(self.base_df.columns)
        if self.has_raw_embeddings:
            self.style_embeddings = _stack_vectors(self.base_df["style_embedding"])
            self.content_embeddings = _stack_vectors(self.base_df["content_embedding"])
        else:
            self.style_embeddings = None
            self.content_embeddings = None

    def get_reducer_options(self) -> list[dict[str, object]]:
        if not self.has_raw_embeddings:
            return [{"label": "Legacy Projection", "value": "legacy"}]
        return get_reducer_options()

    def get_default_reducer(self) -> str:
        if not self.has_raw_embeddings:
            return "legacy"
        return get_default_reducer()

    def build_spec(
        self,
        reducer_name: str,
        *,
        random_state: int = 42,
        parameters: Mapping[str, object] | None = None,
    ) -> ProjectionSpec:
        if reducer_name == "legacy":
            return ProjectionSpec.legacy()
        return ProjectionSpec.for_reducer(
            reducer_name,
            num_samples=len(self.base_df),
            random_state=random_state,
            parameters=parameters,
        )

    def get_projected_dataframe(self, spec: ProjectionSpec):
        digest = spec.digest(self.embedding_digest)
        if digest in self._dataframes:
            return self._dataframes[digest]
        if spec.name == "legacy":
            projected = self.base_df.copy()
            self._dataframes[digest] = projected
            return projected
        if not self.has_raw_embeddings:
            raise ValueError("Raw embeddings are required for a new projection")

        expected_metadata = spec.metadata(self.embedding_digest)
        cache_path = self.cache_root / f"{digest}.npz"
        bundle = load_projection_bundle(cache_path)
        if bundle is None:
            bundle = compute_projection_bundle(
                self.style_embeddings,
                self.content_embeddings,
                spec,
            )
            save_projection_bundle(
                cache_path,
                style_xy=bundle["style_xy"],
                content_xy=bundle["content_xy"],
                metadata=expected_metadata,
            )
        elif bundle["metadata"] != expected_metadata:
            raise ValueError(f"Projection cache metadata mismatch: {cache_path}")

        style_xy = np.asarray(bundle["style_xy"], dtype=np.float32)
        content_xy = np.asarray(bundle["content_xy"], dtype=np.float32)
        expected_shape = (len(self.base_df), 2)
        if style_xy.shape != expected_shape or content_xy.shape != expected_shape:
            raise ValueError(
                f"Projection cache has incompatible coordinates: {cache_path}"
            )
        if not np.isfinite(style_xy).all() or not np.isfinite(content_xy).all():
            raise ValueError(
                f"Projection cache contains non-finite values: {cache_path}"
            )

        projected = self.base_df.copy()
        projected[["x1", "y1"]] = style_xy
        projected[["x2", "y2"]] = content_xy
        self._dataframes[digest] = projected
        return projected


__all__ = [
    "DEFAULT_REDUCER",
    "ProjectionManager",
    "ProjectionSpec",
    "load_projection_bundle",
    "save_projection_bundle",
]
