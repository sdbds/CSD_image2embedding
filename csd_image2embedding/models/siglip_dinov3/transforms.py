"""Square image transforms matching the corrected model processors."""

from __future__ import annotations

import os

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

_pil_limit = os.environ.get("PIL_MAX_IMAGE_PIXELS")
if _pil_limit is not None:
    Image.MAX_IMAGE_PIXELS = None if _pil_limit == "0" else int(_pil_limit)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SIGLIP2_MEAN = [0.5, 0.5, 0.5]
SIGLIP2_STD = [0.5, 0.5, 0.5]


def build_image_transform(
    image_size: int,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Convert, directly resize to a square, and normalize one image."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


_build_image_transform = build_image_transform
