import os
from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.model_smoke
def test_local_siglip_dinov3_modes_smoke(request):
    if not request.config.getoption("--run-model-smoke"):
        pytest.skip("pass --run-model-smoke to load local model assets")

    import numpy as np

    from csd_image2embedding.models.siglip_dinov3.backend import SiglipDinoBackend

    config = Path("configs/siglip_dinov3.yaml").resolve()
    assert config.is_file(), f"Missing model config: {config}"
    device = os.environ.get("CSD_MODEL_SMOKE_DEVICE") or None
    backend = SiglipDinoBackend.from_config(
        config,
        mode="image-only",
        device=device,
        precision="fp32",
    )
    image = Image.new("RGB", (311, 173), (41, 97, 181))

    assert tuple(backend.dino_transform(image).shape) == (3, 224, 224)
    assert tuple(backend.siglip_transform(image).shape) == (3, 256, 256)

    first = backend.encode([image])
    second = backend.encode([image])
    assert first.style_embeddings.shape == (1, 1024)
    assert first.content_embeddings.shape == (1, 1024)
    assert np.isfinite(first.style_embeddings).all()
    assert np.isfinite(first.content_embeddings).all()
    np.testing.assert_allclose(first.style_embeddings, second.style_embeddings)
    np.testing.assert_allclose(first.content_embeddings, second.content_embeddings)

    guided_backend = SiglipDinoBackend(
        backend.model,
        backend.dino_transform,
        backend.siglip_transform,
        mode="caption-guided",
        device=backend.device,
        precision="fp32",
        fingerprint=backend.fingerprint,
        preprocessing_fingerprint=backend.preprocessing_fingerprint,
    )
    guided = guided_backend.encode([image], ["a plain blue geometric image"])
    assert guided.style_embeddings.shape == (1, 1024)
    assert guided.content_embeddings.shape == (1, 1024)
    assert np.isfinite(guided.style_embeddings).all()
    assert np.isfinite(guided.content_embeddings).all()
