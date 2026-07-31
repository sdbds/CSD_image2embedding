import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, functional

from csd_image2embedding.models.siglip_dinov3.transforms import (
    build_image_transform,
)


def test_non_square_image_matches_direct_square_bilinear_reference():
    image = Image.new("RGB", (7, 3))
    image.putdata(
        [
            (x * 30, y * 80, (x + y) * 20)
            for y in range(image.height)
            for x in range(image.width)
        ]
    )
    mean = [0.1, 0.2, 0.3]
    std = [0.5, 0.25, 0.2]

    actual = build_image_transform(5, mean, std)(image)
    expected = functional.to_tensor(image)
    expected = functional.resize(
        expected,
        [5, 5],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    expected = functional.normalize(expected, mean, std)

    assert actual.shape == (3, 5, 5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
