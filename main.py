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
from dash_page import make_dash_kmeans
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

    image = Image.fromarray(image[:, :, ::-1]).resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSD Style Embedding and Visualization")
    parser.add_argument("--dataset_path", type=str, default="dataset.lance", help="Path to the dataset file")
    parser.add_argument("--embeddings_path", type=str, default="embeddings.lance", help="Path to save/load embeddings")
    parser.add_argument("--model_name", type=str, default="yuxi-liu-wired/CSD", help="Name of the pretrained model")
    parser.add_argument("--processor_name", type=str, default="openai/clip-vit-large-patch14", help="Name of the processor")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--k_clusters", type=int, default=122, help="Number of clusters for KMeans")
    args = parser.parse_args()

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

        embeddings = []
        imagelist = []
        hashlist = []
        with Progress() as progress:
            task = progress.add_task(
                "[green]Generating embeddings...", total=len(dataloader)
            )
            for data in dataloader:
                for hash, image in data:
                    image = preprocess_image(image)
                    outputs = pipeline(image)
                    style_outputs = outputs["style_output"].squeeze(0)
                    embeddings.append(style_outputs)
                    buffer = io.BytesIO()
                    image = resize_and_remove_borders(image)
                    image.save(buffer, format="JPEG")
                    image_bytes = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    hashlist.append(hash)
                    imagelist.append(image_bytes)
                    progress.update(task, advance=1)

        reducer = umap.UMAP(n_components=2, random_state=42)
        umap_results = reducer.fit_transform(np.array(embeddings))

        new_data = pa.table(
            {
                "hash": pa.array(hashlist),
                "image": pa.array(imagelist),
                "embeddings": pa.array(embeddings),
                "x": pa.array(umap_results[:, 0]),
                "y": pa.array(umap_results[:, 1]),
            }
        )

        embeddingslance = lance.write_dataset(new_data, args.embeddings_path)

    make_dash_kmeans(embeddingslance, "style", k=args.k_clusters)
