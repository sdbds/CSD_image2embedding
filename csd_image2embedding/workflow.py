"""Top-level orchestration for embedding generation and visualization."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_env() -> None:
    """Disable unused Transformers TensorFlow imports before model loading."""

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


def _collate_valid(batch):
    return [item for item in batch if item is not None]


def _preview_base64(image) -> str:
    import base64
    import io

    preview = image.convert("RGB").copy()
    preview.thumbnail((192, 192))
    buffer = io.BytesIO()
    preview.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run(args) -> int:
    """Run the current end-to-end workflow after lightweight CLI validation."""

    configure_runtime_env()

    import lance
    import numpy as np
    import torch

    from .data.discovery import discover_directory
    from .data.lance import (
        LanceImageDataset,
        build_embedding_table,
        write_source_snapshot,
    )
    from .models.base import validate_backend_mode
    from .models.csd import CSDClipBackend

    supported_modes = {
        "csd": frozenset({"image-only"}),
        "siglip-dinov3": frozenset({"image-only", "caption-guided"}),
    }
    validate_backend_mode(args.backend, supported_modes[args.backend], args.text_mode)
    if args.backend != "csd":
        raise RuntimeError(
            "The corrected SigLIP2-DINOv3 backend is not available in this build yet"
        )

    if not args.dataset_path.exists():
        snapshot = discover_directory(args.train_data_dir)
        write_source_snapshot(snapshot, args.dataset_path)

    backend = CSDClipBackend.from_pretrained(
        args.model_name,
        args.processor_name,
        precision=args.precision,
    )
    source_dataset = LanceImageDataset(args.dataset_path)
    embeddings_path = args.embeddings_path or Path(f"embeddings_{args.backend}.lance")

    if embeddings_path.exists() and not args.rebuild:
        embedding_dataset = lance.dataset(embeddings_path)
    else:
        from torch.utils.data import DataLoader

        try:
            import umap
        except ImportError as error:
            raise ImportError("Embedding generation requires umap-learn") from error

        data_loader = DataLoader(
            source_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=_collate_valid,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )
        styles = []
        contents = []
        paths = []
        previews = []
        for batch_index, batch in enumerate(data_loader):
            if not batch:
                continue
            batch_paths, images, captions = zip(*batch, strict=True)
            try:
                encoded = backend.encode(images, captions)
            except Exception as error:
                first = batch_index * args.batch_size
                last = first + len(batch) - 1
                raise RuntimeError(
                    f"Backend '{args.backend}' failed for batch {first}:{last}; "
                    f"first input: {batch_paths[0]}"
                ) from error
            styles.append(encoded.style_embeddings)
            contents.append(encoded.content_embeddings)
            paths.extend(batch_paths)
            previews.extend(_preview_base64(image) for image in images)

        if not styles:
            raise ValueError("The input dataset contains no readable images")
        style_embeddings = np.concatenate(styles).astype(np.float32, copy=False)
        content_embeddings = np.concatenate(contents).astype(np.float32, copy=False)
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=42)
        style_projection = reducer.fit_transform(style_embeddings)
        content_projection = reducer.fit_transform(content_embeddings)
        table = build_embedding_table(
            paths,
            previews,
            style_embeddings,
            content_embeddings,
            style_projection,
            content_projection,
        )
        mode = "overwrite" if embeddings_path.exists() else "create"
        embedding_dataset = lance.write_dataset(table, embeddings_path, mode=mode)

    from dash_page import make_multi_view_dash
    from view_specs import build_default_view_specs

    titles, parameter_sets = build_default_view_specs(
        args.backend.upper(), args.k_clusters
    )
    make_multi_view_dash(embedding_dataset, titles, parameter_sets, args)
    return 0
