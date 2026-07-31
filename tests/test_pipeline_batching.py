import unittest

import numpy as np
import torch
from PIL import Image

from csd_image2embedding.models.csd import CSDClipPipeline, stack_transformed_images


class DummyCSDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = type("Backbone", (), {})()
        self.backbone.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1)

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        features = torch.arange(
            batch * 4, dtype=pixel_values.dtype, device=pixel_values.device
        ).reshape(batch, 4)
        return features, features + 100, features + 200


class DummyProcessor:
    def __call__(self, images, return_tensors="pt", padding=True, truncation=True):
        batch = len(images)
        pixels = torch.ones((batch, 3, 2, 2), dtype=torch.float32)
        return {"pixel_values": pixels}


class CSDPipelineBatchingTests(unittest.TestCase):
    def test_returns_batched_numpy_outputs_for_multiple_images(self):
        pipeline = CSDClipPipeline(
            model=DummyCSDModel(),
            processor=DummyProcessor(),
            device="cpu",
        )
        images = [Image.new("RGB", (8, 8), "red"), Image.new("RGB", (8, 8), "blue")]

        outputs = pipeline(images)

        self.assertEqual(outputs["style_output"].shape, (2, 4))
        self.assertEqual(outputs["content_output"].shape, (2, 4))
        self.assertEqual(outputs["features"].shape, (2, 4))
        np.testing.assert_allclose(
            outputs["style_output"][0], np.array([200, 201, 202, 203], dtype=np.float32)
        )


class StackTransformedImagesTests(unittest.TestCase):
    def test_stacks_transform_outputs_into_a_batch_tensor(self):
        def identity_transform(image):
            return torch.full((3, 2, 2), image.size[0], dtype=torch.float32)

        batch = stack_transformed_images(
            [Image.new("RGB", (8, 8), "red"), Image.new("RGB", (16, 16), "blue")],
            identity_transform,
            device="cpu",
        )

        self.assertEqual(tuple(batch.shape), (2, 3, 2, 2))
        np.testing.assert_allclose(
            batch[0].cpu().numpy(), np.full((3, 2, 2), 8.0, dtype=np.float32)
        )
        np.testing.assert_allclose(
            batch[1].cpu().numpy(), np.full((3, 2, 2), 16.0, dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
