import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from csd_image2embedding.models.siglip_dinov3 import feature_extractors
from csd_image2embedding.models.siglip_dinov3.feature_extractors import (
    FrozenDINOv3,
    FrozenSigLIP2,
)


class FakeMetaDINO(nn.Module):
    def __init__(
        self,
        feature_dim: int = 4,
        register_tokens: int = 4,
        output_dim: int | None = None,
        *,
        finite: bool = True,
        with_rope: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim or feature_dim
        self.scale = nn.Parameter(torch.ones(()))
        self.storage_tokens = nn.Parameter(torch.zeros(1, register_tokens, feature_dim))
        if with_rope:
            self.rope_embed = SimpleNamespace(periods=torch.ones(1))
        self.finite = finite

    def forward_features(self, pixel_values):
        batch_size = pixel_values.shape[0]
        value = 1.0 if self.finite else float("nan")
        return {
            "x_norm_clstoken": torch.full(
                (batch_size, self.output_dim), value, device=pixel_values.device
            )
            * self.scale
        }


class FakeHFModel(nn.Module):
    def __init__(self, model_type: str, feature_dim: int = 4, registers: int = 4):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type,
            hidden_size=feature_dim,
            num_register_tokens=registers,
        )

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        return SimpleNamespace(
            last_hidden_state=torch.zeros(
                batch_size, 5, self.config.hidden_size, device=pixel_values.device
            )
        )


def test_meta_repo_resolver_supports_torch_without_calling_fn(monkeypatch, tmp_path):
    calls = []

    def resolver(
        github,
        force_reload,
        trust_repo,
        verbose=True,
        skip_validation=False,
    ):
        calls.append((github, force_reload, trust_repo, verbose, skip_validation))
        return tmp_path

    monkeypatch.setattr(torch.hub, "_get_cache_or_reload", resolver)

    assert feature_extractors._resolve_meta_repo("owner/repo:revision") == tmp_path
    assert calls == [("owner/repo:revision", False, True, True, False)]


def test_meta_repo_resolver_supplies_legacy_calling_fn(monkeypatch, tmp_path):
    calls = []

    def resolver(
        github,
        force_reload,
        trust_repo,
        calling_fn,
        verbose=True,
        skip_validation=False,
    ):
        calls.append(
            (
                github,
                force_reload,
                trust_repo,
                calling_fn,
                verbose,
                skip_validation,
            )
        )
        return tmp_path

    monkeypatch.setattr(torch.hub, "_get_cache_or_reload", resolver)

    assert feature_extractors._resolve_meta_repo("owner/repo:revision") == tmp_path
    assert calls == [("owner/repo:revision", False, True, "load", True, False)]


def test_local_pth_dispatches_to_official_meta_loader_and_hashes_checkpoint(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "dinov3_vitl16.pth"
    checkpoint.write_bytes(b"local checkpoint")
    calls = []

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return FakeMetaDINO()

    monkeypatch.setattr(feature_extractors, "_load_meta_dinov3", fake_loader)
    model = FrozenDINOv3(
        str(checkpoint),
        hub_model="dinov3_vitl16",
        hub_repo="facebookresearch/dinov3:test",
        expected_dim=4,
    )

    assert calls == [
        {
            "weights": str(checkpoint.resolve()),
            "hub_model": "dinov3_vitl16",
            "hub_repo": "facebookresearch/dinov3:test",
        }
    ]
    assert (
        model.provenance["checkpoint_sha256"]
        == hashlib.sha256(b"local checkpoint").hexdigest()
    )


def test_meta_loader_uses_strict_state_dict_without_importing_hubconf(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "dinov3.pth"
    checkpoint.write_bytes(b"weights")
    state_calls = []
    expected = SimpleNamespace(
        load_state_dict=lambda state, strict: state_calls.append((state, strict))
    )
    raw_state = {"weight": torch.ones(1)}
    monkeypatch.setattr(feature_extractors, "_resolve_meta_repo", lambda repo: tmp_path)
    monkeypatch.setattr(
        feature_extractors.importlib,
        "import_module",
        lambda name: SimpleNamespace(dinov3_vitl16=lambda pretrained: expected),
    )
    monkeypatch.setattr(
        torch.hub,
        "load",
        lambda *args, **kwargs: pytest.fail("hubconf must not be imported"),
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: raw_state)

    loaded = feature_extractors._load_meta_dinov3(
        weights=str(checkpoint),
        hub_model="dinov3_vitl16",
        hub_repo="facebookresearch/dinov3:test",
    )

    assert loaded is expected
    assert state_calls == [(raw_state, True)]


def test_hugging_face_backend_requires_native_dinov3_and_four_registers(
    monkeypatch,
):
    monkeypatch.setattr(
        feature_extractors.AutoModel,
        "from_pretrained",
        lambda model_id: FakeHFModel("dinov2"),
    )
    with pytest.raises(ValueError, match="dinov3_vit"):
        FrozenDINOv3("wrong/model", expected_dim=4)

    monkeypatch.setattr(
        feature_extractors.AutoModel,
        "from_pretrained",
        lambda model_id: FakeHFModel("dinov3_vit", registers=0),
    )
    with pytest.raises(ValueError, match="4 register tokens"):
        FrozenDINOv3("wrong/registers", expected_dim=4)


@pytest.mark.parametrize(
    ("fake_model", "message"),
    [
        (FakeMetaDINO(register_tokens=0), "4 register tokens"),
        (FakeMetaDINO(with_rope=False), "RoPE"),
    ],
)
def test_meta_backend_requires_register_tokens_and_rope(
    tmp_path, monkeypatch, fake_model, message
):
    checkpoint = tmp_path / "dinov3.pth"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(
        feature_extractors, "_load_meta_dinov3", lambda **kwargs: fake_model
    )

    with pytest.raises(ValueError, match=message):
        FrozenDINOv3(str(checkpoint), expected_dim=4)


@pytest.mark.parametrize(
    ("fake_model", "message"),
    [
        (FakeMetaDINO(output_dim=3), "dimension 4"),
        (FakeMetaDINO(finite=False), "non-finite"),
    ],
)
def test_dinov3_forward_rejects_wrong_or_nonfinite_cls_output(
    tmp_path, monkeypatch, fake_model, message
):
    checkpoint = tmp_path / "dinov3.pth"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setattr(
        feature_extractors, "_load_meta_dinov3", lambda **kwargs: fake_model
    )
    model = FrozenDINOv3(str(checkpoint), expected_dim=4)

    with pytest.raises(RuntimeError, match=message):
        model(torch.ones(1, 3, 2, 2))


def test_siglip_loads_tokenizer_without_constructing_an_image_processor(monkeypatch):
    fake_model = nn.Linear(1, 1)
    fake_tokenizer = object()
    monkeypatch.setattr(
        feature_extractors.AutoModel,
        "from_pretrained",
        lambda model_id: fake_model,
    )
    monkeypatch.setattr(
        feature_extractors.AutoTokenizer,
        "from_pretrained",
        lambda model_id: fake_tokenizer,
    )

    model = FrozenSigLIP2("local/siglip2")

    assert model.tokenizer is fake_tokenizer
    assert not hasattr(model, "processor")
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_upstream_record_matches_all_vendored_file_hashes():
    package_root = (
        Path(__file__).parents[2] / "csd_image2embedding" / "models" / "siglip_dinov3"
    )
    document = (package_root / "UPSTREAM.md").read_text(encoding="utf-8")
    metadata_text = document.split("```json\n", 1)[1].split("\n```", 1)[0]
    metadata = json.loads(metadata_text)

    for relative_path, expected_sha256 in metadata["vendored_sha256"].items():
        actual = hashlib.sha256((package_root / relative_path).read_bytes()).hexdigest()
        assert actual == expected_sha256, relative_path
