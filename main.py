import os
import lance
import torch
import io
import base64
from model import CSD_CLIP
from transformers import CLIPProcessor
from pipeline import CSDCLIPPipeline
from datasets import CustomDataset
from rich.progress import Progress
from dash_page import make_dash_kmeans, make_multi_view_dash
from lancedatasets import transform2lance
from PIL import Image
import numpy as np

import pyarrow as pa
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

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        transform2lance(args.train_data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CSD_CLIP.from_pretrained(args.model_name)
    model.to(device)

    processor = CLIPProcessor.from_pretrained(args.processor_name)
    pipeline = CSDCLIPPipeline(model=model, processor=processor, device=device)

    dataset = CustomDataset(args.dataset_path)

    if os.path.exists(args.embeddings_path):
        embeddingslance = lance.dataset(args.embeddings_path)
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
                for path, image in data:
                    image = preprocess_image(image)
                    outputs = pipeline(image)
                    style_outputs = outputs["style_output"].squeeze(0)
                    content_outputs = outputs["content_output"].squeeze(0)
                    style_embeddings.append(style_outputs)
                    content_embeddings.append(content_outputs)
                    buffer = io.BytesIO()
                    image = resize_and_remove_borders(image)
                    image.save(buffer, format="JPEG")
                    image_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    pathlist.append(path)
                    imagelist.append(image_bytes)
                progress.update(task, advance=1)

        print("Embeddings generated successfully!")
        print("Saving embeddings to disk...")
        reducer = umap.UMAP(
            n_components=2,
            metric="cosine",
            random_state=42,
        )
        style_umap_results = reducer.fit_transform(np.array(style_embeddings))
        content_umap_results = reducer.fit_transform(np.array(content_embeddings))

        new_data = pa.table(
            {
                "path": pa.array(pathlist),
                "image": pa.array(imagelist),
                "x1": pa.array(style_umap_results[:, 0]),
                "y1": pa.array(style_umap_results[:, 1]),
                "x2": pa.array(content_umap_results[:, 0]),
                "y2": pa.array(content_umap_results[:, 1]),
            }
        )

        embeddingslance = lance.write_dataset(new_data, args.embeddings_path)

    titles = [
        "KMeans_style",
        "HDBSCAN_style",
        "KMeans_content",
        "HDBSCAN_content",
    ]
    params_list = [
        {"k": args.k_clusters, "hdbscan": False, "feature_set": "1"},
        {"k": args.k_clusters, "hdbscan": True, "feature_set": "1"},
        {"k": args.k_clusters, "hdbscan": False, "feature_set": "2"},
        {"k": args.k_clusters, "hdbscan": True, "feature_set": "2"},
    ]
    make_multi_view_dash(embeddingslance, titles, params_list, args)
    # make_dash_kmeans(embeddingslance, "style", k=args.k_clusters, output_dir=args.style_ouput_dir)
