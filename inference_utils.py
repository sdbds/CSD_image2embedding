import torch
from PIL import Image


def normalize_image_batch(images):
    if isinstance(images, (str, Image.Image)):
        return [images], True
    return list(images), False


def stack_transformed_images(images, transform, device):
    transformed = [transform(image.convert("RGB")) for image in images]
    return torch.stack(transformed, dim=0).to(device)
