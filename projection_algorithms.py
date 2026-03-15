from functools import lru_cache
import importlib

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from projection_cache import DEFAULT_REDUCER


REDUCER_SPECS = (
    ("pacmap", "PaCMAP", "pacmap"),
    ("umap", "UMAP", "umap"),
    ("tsne", "t-SNE", None),
    ("ivis", "IVIS", "ivis"),
)


def _safe_2d_identity(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(vectors) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(vectors) == 1:
        return np.zeros((1, 2), dtype=np.float32)
    return None


@lru_cache(maxsize=None)
def get_reducer_availability():
    availability = {}
    for name, _, module_name in REDUCER_SPECS:
        if module_name is None:
            availability[name] = {"available": True, "reason": ""}
            continue
        try:
            importlib.import_module(module_name)
            availability[name] = {"available": True, "reason": ""}
        except Exception as exc:
            availability[name] = {"available": False, "reason": str(exc)}
    return availability


def get_reducer_options():
    availability = get_reducer_availability()
    options = []
    for name, label, _ in REDUCER_SPECS:
        option = {"label": label, "value": name}
        if not availability[name]["available"]:
            option["label"] = f"{label} (Unavailable)"
            option["disabled"] = True
        options.append(option)
    return options


def get_default_reducer():
    availability = get_reducer_availability()
    if availability[DEFAULT_REDUCER]["available"]:
        return DEFAULT_REDUCER

    for name, _, _ in REDUCER_SPECS:
        if availability[name]["available"]:
            return name
    return "legacy"


def ensure_reducer_available(reducer_name):
    availability = get_reducer_availability()
    if reducer_name not in availability:
        raise ValueError(f"Unknown reducer: {reducer_name}")
    if not availability[reducer_name]["available"]:
        raise RuntimeError(
            f"Reducer '{reducer_name}' is unavailable: {availability[reducer_name]['reason']}"
        )


def _resolve_tsne_perplexity(num_samples):
    if num_samples <= 2:
        return 1
    return min(30, max(2, num_samples // 3))


def _compute_umap(vectors, random_state):
    import umap

    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


def _compute_pacmap(vectors, random_state):
    import pacmap

    reducer = pacmap.PaCMAP(
        n_components=2,
        distance="angular",
        random_state=random_state,
    )
    return reducer.fit_transform(vectors, init="pca")


def _compute_tsne(vectors, random_state):
    reducer = TSNE(
        n_components=2,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        perplexity=_resolve_tsne_perplexity(len(vectors)),
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


def _compute_ivis(vectors, random_state):
    del random_state  # ivis does not expose sklearn-style random_state in its basic API
    from ivis import Ivis

    scaled = MinMaxScaler().fit_transform(vectors)
    model = Ivis(
        embedding_dims=2,
        k=min(15, len(vectors) - 1),
        knn_distance_metric="angular",
        model="maaten",
        n_epochs_without_progress=20,
        verbose=0,
    )
    return model.fit_transform(scaled)


def compute_2d_projection(vectors, reducer_name, random_state=42):
    vectors = np.asarray(vectors, dtype=np.float32)
    trivial = _safe_2d_identity(vectors)
    if trivial is not None:
        return trivial

    if reducer_name == "legacy":
        raise ValueError("Legacy projection must come from stored coordinates.")

    ensure_reducer_available(reducer_name)

    if reducer_name == "umap":
        projected = _compute_umap(vectors, random_state)
    elif reducer_name == "pacmap":
        projected = _compute_pacmap(vectors, random_state)
    elif reducer_name == "tsne":
        projected = _compute_tsne(vectors, random_state)
    elif reducer_name == "ivis":
        projected = _compute_ivis(vectors, random_state)
    else:
        raise ValueError(f"Unknown reducer: {reducer_name}")

    return np.asarray(projected, dtype=np.float32)


def compute_projection_bundle(
    style_embeddings,
    content_embeddings,
    reducer_name,
    random_state=42,
):
    return {
        "style_xy": compute_2d_projection(
            style_embeddings,
            reducer_name,
            random_state=random_state,
        ),
        "content_xy": compute_2d_projection(
            content_embeddings,
            reducer_name,
            random_state=random_state,
        ),
    }
