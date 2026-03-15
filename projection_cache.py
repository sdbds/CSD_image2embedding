import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_REDUCER = "pacmap"
CACHE_VERSION = 1


def build_dataset_fingerprint(paths, style_embeddings, content_embeddings):
    hasher = hashlib.sha256()
    hasher.update(str(CACHE_VERSION).encode("utf-8"))
    hasher.update(str(len(paths)).encode("utf-8"))

    for path in paths:
        hasher.update(path.encode("utf-8"))
        hasher.update(b"\0")

    for name, vectors in (
        ("style", style_embeddings),
        ("content", content_embeddings),
    ):
        array = np.asarray(vectors, dtype=np.float32)
        hasher.update(name.encode("utf-8"))
        hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        if array.size:
            hasher.update(array[0].tobytes())
            hasher.update(array[-1].tobytes())
            mean_vector = array.mean(axis=0, dtype=np.float64).astype(np.float32)
            hasher.update(mean_vector.tobytes())

    return hasher.hexdigest()[:16]


def build_projection_cache_path(cache_root, dataset_fingerprint, reducer_name):
    cache_root = Path(cache_root)
    return cache_root / dataset_fingerprint / f"{reducer_name}.npz"


def save_projection_bundle(
    cache_root,
    dataset_fingerprint,
    reducer_name,
    style_xy,
    content_xy,
    metadata=None,
):
    cache_path = build_projection_cache_path(
        cache_root=cache_root,
        dataset_fingerprint=dataset_fingerprint,
        reducer_name=reducer_name,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "style_xy": np.asarray(style_xy, dtype=np.float32),
        "content_xy": np.asarray(content_xy, dtype=np.float32),
        "metadata_json": np.array(json.dumps(metadata or {})),
    }
    np.savez(cache_path, **payload)
    return cache_path


def load_projection_bundle(cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as bundle:
        metadata_json = str(bundle["metadata_json"].item())
        return {
            "style_xy": bundle["style_xy"].astype(np.float32),
            "content_xy": bundle["content_xy"].astype(np.float32),
            "metadata": json.loads(metadata_json),
        }
