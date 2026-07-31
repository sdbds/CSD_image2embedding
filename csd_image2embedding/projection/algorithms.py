"""Lazy two-dimensional projection implementations."""

from __future__ import annotations

import importlib
from functools import cache
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

DEFAULT_REDUCER = "pacmap"
REDUCER_SPECS = (
    ("pacmap", "PaCMAP", "pacmap", "pacmap"),
    ("umap", "UMAP", "umap", "umap-learn"),
    ("tsne", "t-SNE", None, "scikit-learn"),
    ("ivis", "IVIS", "ivis", "ivis"),
)


def _safe_2d_identity(vectors: np.ndarray) -> np.ndarray | None:
    if len(vectors) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(vectors) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    return None


@cache
def get_reducer_availability() -> dict[str, dict[str, object]]:
    availability = {}
    for name, _, module_name, _ in REDUCER_SPECS:
        if module_name is None:
            availability[name] = {"available": True, "reason": ""}
            continue
        try:
            importlib.import_module(module_name)
            availability[name] = {"available": True, "reason": ""}
        except ImportError as error:
            availability[name] = {"available": False, "reason": str(error)}
    return availability


def get_reducer_options() -> list[dict[str, object]]:
    availability = get_reducer_availability()
    options = []
    for name, label, _, _ in REDUCER_SPECS:
        option: dict[str, object] = {"label": label, "value": name}
        if not availability[name]["available"]:
            option["label"] = f"{label} (Unavailable)"
            option["disabled"] = True
        options.append(option)
    return options


def get_default_reducer() -> str:
    availability = get_reducer_availability()
    if availability[DEFAULT_REDUCER]["available"]:
        return DEFAULT_REDUCER
    for name, _, _, _ in REDUCER_SPECS:
        if availability[name]["available"]:
            return name
    return "legacy"


def ensure_reducer_available(reducer_name: str) -> None:
    availability = get_reducer_availability()
    if reducer_name not in availability:
        raise ValueError(f"Unknown reducer: {reducer_name}")
    if not availability[reducer_name]["available"]:
        reason = availability[reducer_name]["reason"]
        raise RuntimeError(f"Reducer '{reducer_name}' is unavailable: {reason}")


def get_implementation_version(reducer_name: str) -> str:
    if reducer_name == "legacy":
        return "builtin"
    distribution = next(
        (
            distribution
            for name, _, _, distribution in REDUCER_SPECS
            if name == reducer_name
        ),
        None,
    )
    if distribution is None:
        raise ValueError(f"Unknown reducer: {reducer_name}")
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def default_parameters(reducer_name: str, num_samples: int) -> dict[str, object]:
    if reducer_name == "pacmap":
        return {"n_components": 2, "distance": "angular", "init": "pca"}
    if reducer_name == "umap":
        return {"n_components": 2, "metric": "cosine"}
    if reducer_name == "tsne":
        perplexity = 1 if num_samples <= 2 else min(30, max(2, num_samples // 3))
        return {
            "n_components": 2,
            "metric": "cosine",
            "init": "pca",
            "learning_rate": "auto",
            "perplexity": perplexity,
        }
    if reducer_name == "ivis":
        return {
            "embedding_dims": 2,
            "k": max(1, min(15, num_samples - 1)),
            "knn_distance_metric": "angular",
            "model": "maaten",
            "n_epochs_without_progress": 20,
            "verbose": 0,
        }
    if reducer_name == "legacy":
        return {}
    raise ValueError(f"Unknown reducer: {reducer_name}")


def compute_2d_projection(vectors, spec) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Projection input must have rank 2")
    trivial = _safe_2d_identity(vectors)
    if trivial is not None:
        return trivial
    if spec.name == "legacy":
        raise ValueError("Legacy projection must come from stored coordinates")
    ensure_reducer_available(spec.name)

    parameters = dict(spec.parameters)
    if "random_state" in parameters:
        raise ValueError("Projection parameters must not duplicate random_state")
    if spec.name == "umap":
        import umap

        reducer = umap.UMAP(random_state=spec.random_state, **parameters)
        projected = reducer.fit_transform(vectors)
    elif spec.name == "pacmap":
        import pacmap

        init = parameters.pop("init", "pca")
        reducer = pacmap.PaCMAP(random_state=spec.random_state, **parameters)
        projected = reducer.fit_transform(vectors, init=init)
    elif spec.name == "tsne":
        reducer = TSNE(random_state=spec.random_state, **parameters)
        projected = reducer.fit_transform(vectors)
    elif spec.name == "ivis":
        from ivis import Ivis

        scaled = MinMaxScaler().fit_transform(vectors)
        projected = Ivis(**parameters).fit_transform(scaled)
    else:
        raise ValueError(f"Unknown reducer: {spec.name}")
    return np.asarray(projected, dtype=np.float32)


def compute_projection_bundle(style_embeddings, content_embeddings, spec):
    return {
        "style_xy": compute_2d_projection(style_embeddings, spec),
        "content_xy": compute_2d_projection(content_embeddings, spec),
    }
