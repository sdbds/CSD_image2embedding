import os
import lance
import torch
import io
import base64
from runtime_env import configure_runtime_env

configure_runtime_env()

from model import CSD_CLIP
from transformers import CLIPProcessor
from pipeline import CSDCLIPPipeline
from datasets import CustomDataset
from rich.progress import Progress
from dash_page import make_dash_kmeans, make_multi_view_dash
from lancedatasets import transform2lance
from embedding_storage import build_embedding_table
from precision_utils import resolve_amp_dtype
from view_specs import build_default_view_specs
from PIL import Image
import numpy as np

import umap
import argparse

IMAGE_SIZE = 336


def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def resize_image(image, max_resolution=192):
    if max(image.width, image.height) > max_resolution:
        image = image.resize(
            (max_resolution, int(image.height * max_resolution / image.width))
        )
    return image


def remove_white_borders(image):
    image_np = np.array(image)
    mask = image_np != 255
    coords = np.argwhere(mask)
    x0, y0, _ = coords.min(axis=0)
    x1, y1, _ = coords.max(axis=0) + 1  # slices are exclusive at the top
    cropped_image = image_np[x0:x1, y0:y1, :]
    return Image.fromarray(cropped_image)


def resize_and_remove_borders(image, max_resolution=192):
    image = resize_image(image, max_resolution)
    image = remove_white_borders(image)
    return image


def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]  # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(
        image,
        ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)),
        mode="constant",
        constant_values=255,
    )

    image = Image.fromarray(image[:, :, ::-1]).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
    )

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSD Style Embedding and Visualization"
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="datasets",
        help="directory for train images",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets.lance",
        help="Path to the dataset file",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="embeddings.lance",
        help="Path to save/load embeddings",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="yuxi-liu-wired/CSD",
        help="Name of the pretrained model",
    )
    parser.add_argument(
        "--processor_name",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Name of the processor",
    )
    parser.add_argument(
        "--batch_size", type=int, default=12, help="Batch size for data loading"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--k_clusters", type=int, default=40, help="Number of clusters for KMeans"
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=10,
        help="smaller size get more clusters for HDBSCAN",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for the classified images",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying images",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="csd",
        choices=["csd", "sd"],
        help="Backend model: 'csd' (default) or 'sd' (StyleDecoupler DINOv3+SigLIP2)",
    )
    parser.add_argument(
        "--sd_config",
        type=str,
        default="sd_config.yaml",
        help="Path to sd_config.yaml (used when --model_type=sd)",
    )
    parser.add_argument(
        "--sd_checkpoint",
        type=str,
        default=None,
        help="Override checkpoint_path in sd_config.yaml (optional)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Inference precision. 'auto' uses fp16 on CUDA and fp32 otherwise.",
    )
    parser.add_argument(
        "--finch_partition_index",
        type=int,
        default=1,
        help="FINCH hierarchy partition index. 1 is the default, 0 is the finest partition.",
    )

    args = parser.parse_args()

    # Append model_type suffix to embeddings_path to avoid overwriting
    if args.embeddings_path == "embeddings.lance":
        args.embeddings_path = f"embeddings_{args.model_type}.lance"

    if not os.path.exists(args.dataset_path):
        transform2lance(args.train_data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = resolve_amp_dtype(device, args.precision)

    if args.model_type == "csd":
        model = CSD_CLIP.from_pretrained(args.model_name)
        model.to(device)
        processor = CLIPProcessor.from_pretrained(args.processor_name)
        pipeline = CSDCLIPPipeline(
            model=model,
            processor=processor,
            device=device,
            amp_dtype=amp_dtype,
        )
    else:
        from sd_pipeline import StyleDecouplerPipeline
        pipeline = StyleDecouplerPipeline.from_config(
            args.sd_config,
            args.sd_checkpoint,
            device=device,
            amp_dtype=amp_dtype,
        )

    dataset = CustomDataset(args.dataset_path)

    if os.path.exists(args.embeddings_path):
        embeddingslance = lance.dataset(args.embeddings_path)
        schema_names = set(embeddingslance.schema.names)
        if {"style_embedding", "content_embedding"} - schema_names:
            print(
                "Warning: cached embeddings dataset does not contain raw embedding columns."
            )
            print(
                "Clustering will fall back to the stored 2D projection until you regenerate the cache."
            )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn_remove_corrupted,
            drop_last=False,
            pin_memory=True,
        )

        style_embeddings = []
        content_embeddings = []
        imagelist = []
        pathlist = []
        with Progress() as progress:
            task = progress.add_task(
                "[green]Generating embeddings...", total=len(dataloader)
            )
            for data in dataloader:
                paths = []
                model_images = []
                preview_images = []

                for path, image in data:
                    prepared_image = (
                        preprocess_image(image) if args.model_type == "csd" else image
                    )
                    paths.append(path)
                    model_images.append(prepared_image)
                    preview_images.append(prepared_image)

                outputs = pipeline(model_images)
                style_outputs = np.asarray(outputs["style_output"])
                content_outputs = np.asarray(outputs["content_output"])
                style_embeddings.extend(style_outputs)
                content_embeddings.extend(content_outputs)

                for path, image in zip(paths, preview_images):
                    buffer = io.BytesIO()
                    image = resize_and_remove_borders(image)
                    image.save(buffer, format="JPEG")
                    image_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    pathlist.append(path)
                    imagelist.append(image_bytes)
                progress.update(task, advance=1)

        print("Embeddings generated successfully!")
        print("Saving embeddings to disk...")
        style_embedding_array = np.asarray(style_embeddings, dtype=np.float32)
        content_embedding_array = np.asarray(content_embeddings, dtype=np.float32)
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        style_umap_results = reducer.fit_transform(style_embedding_array)
        content_umap_results = reducer.fit_transform(content_embedding_array)

        new_data = build_embedding_table(
            pathlist=pathlist,
            imagelist=imagelist,
            style_embeddings=style_embedding_array,
            content_embeddings=content_embedding_array,
            style_projection=style_umap_results,
            content_projection=content_umap_results,
        )

        embeddingslance = lance.write_dataset(new_data, args.embeddings_path)

    model_label = args.model_type.upper()
    titles, params_list = build_default_view_specs(model_label, args.k_clusters)
    make_multi_view_dash(embeddingslance, titles, params_list, args)
    # make_dash_kmeans(embeddingslance, "style", k=args.k_clusters, output_dir=args.style_ouput_dir)
