import numpy as np
import torch
from PIL import Image

from csd_image2embedding.models.csd import CSDClipBackend, preprocess_csd_image


class FakeCSDModel(torch.nn.Module):
    embedding_dim = 2
    content_dim = 2
    style_dim = 2

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    @property
    def dtype(self):
        return self.weight.dtype

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        features = torch.ones((batch_size, 2), device=pixel_values.device)
        content = torch.tensor([0.0, 2.0], device=pixel_values.device).repeat(
            batch_size, 1
        )
        style = torch.tensor([3.0, 4.0], device=pixel_values.device).repeat(
            batch_size, 1
        )
        return features, content, style


class FakeProcessor:
    def __call__(self, images, return_tensors, padding, truncation):
        del return_tensors, padding, truncation
        return {"pixel_values": torch.ones((len(images), 3, 2, 2))}


def test_csd_backend_returns_normalized_embedding_batch():
    backend = CSDClipBackend(
        model=FakeCSDModel(),
        processor=FakeProcessor(),
        model_name="fake-csd",
        processor_name="fake-processor",
        device="cpu",
        precision="fp32",
    )

    result = backend.encode([Image.new("RGB", (4, 2), "red")])

    assert result.backend == "csd"
    assert result.mode == "image-only"
    assert result.style_embeddings.dtype == np.float32
    np.testing.assert_allclose(result.style_embeddings, [[0.6, 0.8]])
    np.testing.assert_allclose(result.content_embeddings, [[0.0, 1.0]])
    result.validate(expected_rows=1)


def test_csd_backend_accepts_a_single_image_path(tmp_path):
    image_path = tmp_path / "input.png"
    Image.new("RGB", (4, 2), "red").save(image_path)
    backend = CSDClipBackend(
        model=FakeCSDModel(),
        processor=FakeProcessor(),
        model_name="fake-csd",
        processor_name="fake-processor",
        device="cpu",
        precision="fp32",
    )

    result = backend.encode(image_path)

    assert result.style_embeddings.shape == (1, 2)


def test_csd_fingerprint_changes_with_processor_identity():
    common = {
        "model": FakeCSDModel(),
        "processor": FakeProcessor(),
        "model_name": "fake-csd",
        "device": "cpu",
        "precision": "fp32",
    }

    first = CSDClipBackend(processor_name="processor-a", **common)
    second = CSDClipBackend(processor_name="processor-b", **common)

    assert first.fingerprint != second.fingerprint
    assert len(first.preprocessing_fingerprint) == 64
    assert first.preprocessing_fingerprint != second.preprocessing_fingerprint


def test_csd_auto_precision_fingerprint_uses_the_resolved_device_precision():
    common = {
        "model": FakeCSDModel(),
        "processor": FakeProcessor(),
        "model_name": "fake-csd",
        "processor_name": "fake-processor",
        "precision": "auto",
    }

    cpu = CSDClipBackend(device="cpu", **common)
    cuda = CSDClipBackend(device="cuda", **common)

    assert cpu.fingerprint != cuda.fingerprint


def test_csd_preprocessing_pads_non_square_images_to_model_size():
    image = Image.new("RGB", (8, 4), "red")

    prepared = preprocess_csd_image(image)

    assert prepared.size == (336, 336)
    assert min(prepared.getpixel((168, 10))) >= 240
