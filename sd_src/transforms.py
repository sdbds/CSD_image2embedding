"""Shared image transforms and normalisation constants (copied from styledecouple_dinov3)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# ImageNet statistics — used for DINOv3 inputs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# SigLIP2 uses simple 0.5/0.5 normalisation
SIGLIP2_MEAN = [0.5, 0.5, 0.5]
SIGLIP2_STD  = [0.5, 0.5, 0.5]


class ResizeInterArea:
    """Resize the shorter side to `size` using cv2 INTER_AREA."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w <= h:
            new_w, new_h = self.size, round(h * self.size / w)
        else:
            new_w, new_h = round(w * self.size / h), self.size
        arr = np.array(img.convert("RGB"))
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return Image.fromarray(arr)


def build_image_transform(
    image_size: int,
    mean: list[float] = IMAGENET_MEAN,
    std: list[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Resize (cv2 INTER_AREA) → centre-crop → to-tensor → normalise."""
    return transforms.Compose([
        ResizeInterArea(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
