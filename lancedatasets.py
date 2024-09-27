import argparse
import os
from PIL import Image
import hashlib

import lance
import pyarrow as pa

from tqdm.auto import tqdm

IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".WEBP",
    ".BMP",
]

try:
    import pillow_avif

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

# JPEG-XL on Windows
try:
    import pillow_jxl

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

# 将 IMAGE_EXTENSIONS 转换为元组
IMAGE_EXTENSIONS = tuple(IMAGE_EXTENSIONS)


def load_data(images_dir, texts_dir):
    data = []
    if texts_dir:
        images = sorted(os.listdir(images_dir))
        texts = sorted(os.listdir(texts_dir))

        for image_file, text_file in zip(images, texts):
            if image_file.endswith(IMAGE_EXTENSIONS) and text_file.endswith(".txt"):
                with open(
                    os.path.join(texts_dir, text_file), "r", encoding="utf-8"
                ) as file:
                    caption = file.read().strip()
                data.append(
                    {
                        "image_path": os.path.join(root, image_file),
                        "caption": caption,
                    }
                )

    else:
        for root, dirs, files in os.walk(images_dir):
            for image_file in files:
                if image_file.endswith(IMAGE_EXTENSIONS):
                    text_file = os.path.splitext(image_file)[0] + ".txt"
                    text_path = os.path.join(root, text_file)
                    if os.path.exists(text_path):
                        with open(text_path, "r", encoding="utf-8") as file:
                            caption = file.read().strip()

                        data.append(
                            {
                                "image_path": os.path.join(root, image_file),
                                "caption": caption,
                            }
                        )
                    else:
                        data.append(
                            {
                                "image_path": os.path.join(root, image_file),
                                "caption": "",
                            }
                        )
    return data


def process(data):
    for item in tqdm(data):
        image_path = item["image_path"]
        caption = item["caption"]
        print(f"Processing image '{image_path}'...")
        print(f"Caption: {caption}")
        try:
            with open(image_path, "rb") as im:
                binary_im = im.read()
                image_hash = hashlib.sha256(binary_im).hexdigest()
            img = Image.open(image_path)
            width, height = img.size
            image_size = os.path.getsize(image_path)
        except FileNotFoundError:
            print(
                f"Image '{os.path.basename(image_path)}' not found in the folder, skipping."
            )
            continue
        except (IOError, SyntaxError) as e:
            print(
                f"Error opening image '{os.path.basename(image_path)}': {str(e)}. Truncating the file."
            )
            continue

        print(f"Image '{image_path}' processed successfully.")

        filename = pa.array([os.path.abspath(image_path)], type=pa.string())
        extension = pa.array([os.path.splitext(os.path.basename(image_path))[1]], type=pa.string())
        width = pa.array([int(width)], type=pa.int32())
        height = pa.array([int(height)], type=pa.int32())
        size = pa.array([image_size], type=pa.int64())
        hash = pa.array([image_hash], type=pa.string())
        img = pa.array([binary_im], type=pa.binary())
        captions = pa.array([caption], type=pa.string())

        yield pa.RecordBatch.from_arrays(
            [filename, extension, hash, size, width, height, img, captions],
            [
                "filename",
                "extension",
                "hash",
                "size",
                "width",
                "height",
                "image",
                "captions",
            ],
        )


def transform2lance(args: argparse.Namespace) -> None:

    data = load_data(args.train_data_dir, args.captions_dir)

    schema = pa.schema(
        [
            pa.field("filename", pa.string()),
            pa.field("extension", pa.string()),
            pa.field("hash", pa.string()),
            pa.field("size", pa.int64()),
            pa.field("width", pa.int32()),
            pa.field("height", pa.int32()),
            pa.field("image", pa.binary()),
            pa.field("captions", pa.string()),
        ]
    )
    try:
        reader = pa.RecordBatchReader.from_batches(schema, process(data))
        lance.write_dataset(reader, args.output_name + ".lance", schema)
    except AttributeError as e:
        print(f"AttributeError: {e}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("train_data_dir", type=str, help="directory for train images")
    parser.add_argument(
        "--captions_dir", type=str, default=None, help="directory for train images"
    )
    parser.add_argument(
        "--output_name", type=str, default="dataset", help="directory for train images"
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    transform2lance(args)
