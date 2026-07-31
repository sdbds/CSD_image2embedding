from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from csd_image2embedding.models import get_supported_text_modes
from csd_image2embedding.models.base import validate_backend_mode
from csd_image2embedding.models.siglip_dinov3.backend import (
    SiglipDinoBackend,
    caption_guided_embeddings,
    load_siglip_dino_config,
    validate_checkpoint_provenance,
)


class FakeTokenizer:
    def __call__(self, captions, **kwargs):
        assert kwargs == {
            "max_length": 64,
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        }
        return {
            "input_ids": torch.ones((len(captions), 2), dtype=torch.int64),
            "attention_mask": torch.ones((len(captions), 2), dtype=torch.int64),
        }


class FakeDecoupler:
    def __init__(self):
        self.siglip = SimpleNamespace(tokenizer=FakeTokenizer())

    def encode_image_siglip(self, pixels):
        return torch.tensor([3.0, 4.0], device=pixels.device).repeat(len(pixels), 1)

    def encode_image_dino(self, pixels):
        return torch.tensor([0.0, 2.0], device=pixels.device).repeat(len(pixels), 1)

    def encode_text(self, input_ids, attention_mask):
        del attention_mask
        return torch.tensor([1.0, 0.0], device=input_ids.device).repeat(
            len(input_ids), 1
        )


def _fake_backend(mode="image-only"):
    def transform(image):
        del image
        return torch.zeros((3, 2, 2), dtype=torch.float32)

    return SiglipDinoBackend(
        model=FakeDecoupler(),
        dino_transform=transform,
        siglip_transform=transform,
        mode=mode,
        device="cpu",
        precision="fp32",
        fingerprint="fake-model",
    )


def _fake_image():
    return Image.new("RGB", (4, 2), "red")


def test_image_only_returns_siglip_style_and_projected_dino_content():
    output = _fake_backend().encode([_fake_image()])

    np.testing.assert_allclose(output.style_embeddings, [[0.6, 0.8]])
    np.testing.assert_allclose(output.content_embeddings, [[0.0, 1.0]])
    assert output.mode == "image-only"
    assert output.backend == "siglip-dinov3"


def test_caption_guided_uses_caption_as_content_only():
    b_bar = torch.tensor([[0.6, 0.8]])
    c_bar = torch.tensor([[0.0, 1.0]])
    d_bar = torch.tensor([[1.0, 0.0]])

    style, content = caption_guided_embeddings(b_bar, c_bar, d_bar)

    expected_content = torch.nn.functional.normalize(c_bar + d_bar, dim=-1)
    similarity = (b_bar * expected_content).sum(dim=-1, keepdim=True)
    expected_style = torch.nn.functional.normalize(
        b_bar - torch.clamp(1.0 - similarity, min=0.0) * similarity * expected_content,
        dim=-1,
    )
    torch.testing.assert_close(content, expected_content)
    torch.testing.assert_close(style, expected_style)


def test_caption_guided_requires_complete_valid_captions():
    backend = _fake_backend(mode="caption-guided")

    with pytest.raises(ValueError, match=r"valid=1, missing=1, empty=0, unreadable=0"):
        backend.encode([_fake_image(), _fake_image()], ["a lake", None])


def test_invalid_caption_coverage_fails_before_image_inference():
    backend = _fake_backend(mode="caption-guided")
    backend.model.encode_image_siglip = lambda pixels: pytest.fail(
        "image inference must not run"
    )

    with pytest.raises(ValueError, match="missing=1"):
        backend.encode([_fake_image()], [None])


def test_caption_guided_backend_tokenizes_generic_caption_as_content():
    output = _fake_backend(mode="caption-guided").encode(
        [_fake_image()], ["a lake near a mountain"]
    )

    assert output.mode == "caption-guided"
    np.testing.assert_allclose(np.linalg.norm(output.style_embeddings, axis=1), [1.0])
    np.testing.assert_allclose(np.linalg.norm(output.content_embeddings, axis=1), [1.0])


def test_caption_guided_synthesizes_a_missing_siglip_attention_mask():
    backend = _fake_backend(mode="caption-guided")
    backend.model.siglip.tokenizer = lambda captions, **kwargs: {
        "input_ids": torch.ones((len(captions), 4), dtype=torch.int64)
    }
    observed = {}

    def encode_text(input_ids, attention_mask):
        observed["mask"] = attention_mask.detach().cpu()
        return torch.tensor([1.0, 0.0], device=input_ids.device).repeat(
            len(input_ids), 1
        )

    backend.model.encode_text = encode_text

    backend.encode([_fake_image()], ["a plain caption"])

    assert torch.equal(observed["mask"], torch.ones((1, 4), dtype=torch.int64))


@pytest.mark.parametrize(
    ("backend", "mode", "accepted"),
    [
        ("csd", "image-only", True),
        ("csd", "caption-guided", False),
        ("siglip-dinov3", "image-only", True),
        ("siglip-dinov3", "caption-guided", True),
    ],
)
def test_backend_mode_compatibility_matrix(backend, mode, accepted):
    supported = get_supported_text_modes(backend)
    if accepted:
        assert validate_backend_mode(backend, supported, mode) == mode
    else:
        with pytest.raises(ValueError, match=backend):
            validate_backend_mode(backend, supported, mode)


def test_config_expands_environment_and_resolves_paths_from_config(
    tmp_path, monkeypatch
):
    model_root = tmp_path / "models"
    monkeypatch.setenv("STYLE_MODEL_ROOT", str(model_root))
    config_path = tmp_path / "config" / "model.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
model_base_path: "${STYLE_MODEL_ROOT}"
model:
  dino_model_id: "./dinov3.pth"
  dino_hub_model: "dinov3_vitl16"
  dino_hub_repo: "facebookresearch/dinov3:commit"
  siglip_model_id: "./siglip2"
  dino_dim: 4
  siglip_dim: 4
  projector_hidden_dim: 8
  projector_num_layers: 2
  projector_dropout: 0.0
preprocessing:
  dino_image_size: 4
  siglip_image_size: 5
checkpoint_path: "./checkpoints/checkpoint_best.safetensors"
""".strip(),
        encoding="utf-8",
    )

    config = load_siglip_dino_config(config_path)

    assert config.text_max_length == 64
    assert config.dino_model_id == str((model_root / "dinov3.pth").resolve())
    assert config.siglip_model_id == str((model_root / "siglip2").resolve())
    assert (
        config.checkpoint_path
        == (model_root / "checkpoints" / "checkpoint_best.safetensors").resolve()
    )


def test_checkpoint_provenance_ignores_host_paths_but_rejects_model_contract():
    config = SimpleNamespace(
        dino_model_id="D:/models/dinov3.pth",
        dino_hub_model="dinov3_vitl16",
        dino_hub_repo="facebookresearch/dinov3:commit",
        dino_dim=4,
        siglip_dim=4,
        projector_hidden_dim=8,
        projector_num_layers=2,
        projector_dropout=0.0,
        dino_image_size=4,
        siglip_image_size=5,
    )
    provenance = {
        "features": {
            "dino": {
                "backend": "meta",
                "model_ref": "/training-host/dinov3.pth",
                "configured_ref": "./dinov3.pth",
                "checkpoint_sha256": "dino-sha",
                "hub_model": "dinov3_vitl16",
                "hub_repo": "facebookresearch/dinov3:commit",
                "architecture": {
                    "feature_dim": 4,
                    "register_tokens": 4,
                    "positional_encoding": "rope",
                },
            },
            "preprocessing": {
                "dino_image_size": 4,
                "siglip_image_size": 5,
                "resize": "square_bilinear_antialias",
            },
            "dataset": {"source": "/training-host/data"},
        },
        "projector": {
            "input_dim": 4,
            "hidden_dim": 8,
            "output_dim": 4,
            "num_layers": 2,
            "dropout": 0.0,
        },
    }

    validate_checkpoint_provenance(config, provenance, dino_sha256="dino-sha")

    provenance["projector"]["output_dim"] = 5
    with pytest.raises(ValueError, match=r"projector\.output_dim"):
        validate_checkpoint_provenance(config, provenance, dino_sha256="dino-sha")
