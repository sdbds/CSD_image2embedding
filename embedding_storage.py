import numpy as np
import pyarrow as pa


def _to_list_array(vectors):
    return pa.array(
        [np.asarray(vector, dtype=np.float32).tolist() for vector in vectors],
        type=pa.list_(pa.float32()),
    )


def build_embedding_table(
    pathlist,
    imagelist,
    style_embeddings,
    content_embeddings,
    style_projection,
    content_projection,
):
    return pa.table(
        {
            "path": pa.array(pathlist),
            "image": pa.array(imagelist),
            "style_embedding": _to_list_array(style_embeddings),
            "content_embedding": _to_list_array(content_embeddings),
            "x1": pa.array(style_projection[:, 0], type=pa.float32()),
            "y1": pa.array(style_projection[:, 1], type=pa.float32()),
            "x2": pa.array(content_projection[:, 0], type=pa.float32()),
            "y2": pa.array(content_projection[:, 1], type=pa.float32()),
        }
    )
