"""Application backend for corrected SigLIP2 and DINOv3 inference."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

from ..base import EmbeddingBatch, TextMode, validate_backend_mode
from .feature_extractors import (
    DEFAULT_DINOV3_HUB_REPO,
    FrozenDINOv3,
    FrozenSigLIP2,
)
from .projector import AlignmentProjector
from .style_decoupler import StyleDecoupler
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SIGLIP2_MEAN,
    SIGLIP2_STD,
    build_image_transform,
)

BACKEND_SCHEMA_VERSION = 1


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expand_path(value: str | Path, base: Path) -> Path:
    original = str(value)
    expanded = os.path.expandvars(os.path.expanduser(original))
    if "$" in expanded or "%" in expanded:
        raise ValueError(
            f"Path contains an unresolved environment variable: {original}"
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_model_reference(value: str, base: Path) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded or "%" in expanded:
        raise ValueError(f"Model reference has an unresolved variable: {value}")
    path = Path(expanded)
    is_explicit_path = (
        path.is_absolute()
        or value.startswith((".", "~", "$", "%"))
        or (base / path).exists()
    )
    return str(_expand_path(expanded, base)) if is_explicit_path else expanded


@dataclass(frozen=True)
class SiglipDinoConfig:
    config_path: Path
    dino_model_id: str
    dino_hub_model: str
    dino_hub_repo: str
    siglip_model_id: str
    checkpoint_path: Path
    dino_dim: int
    siglip_dim: int
    projector_hidden_dim: int
    projector_num_layers: int
    projector_dropout: float
    dino_image_size: int
    siglip_image_size: int
    dino_mean: tuple[float, float, float]
    dino_std: tuple[float, float, float]
    siglip_mean: tuple[float, float, float]
    siglip_std: tuple[float, float, float]

    def preprocessing_payload(self) -> dict[str, object]:
        return {
            "resize": "square_bilinear_antialias",
            "dino_image_size": self.dino_image_size,
            "siglip_image_size": self.siglip_image_size,
            "dino_mean": list(self.dino_mean),
            "dino_std": list(self.dino_std),
            "siglip_mean": list(self.siglip_mean),
            "siglip_std": list(self.siglip_std),
        }


def _three_floats(value, field_name: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly three values")
    return values


def load_siglip_dino_config(
    path: Path,
    checkpoint_override: Path | None = None,
) -> SiglipDinoConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SigLIP2-DINOv3 config must be a YAML mapping")
    model = payload.get("model")
    preprocessing = payload.get("preprocessing")
    if not isinstance(model, dict) or not isinstance(preprocessing, dict):
        raise ValueError("Config requires 'model' and 'preprocessing' mappings")

    config_directory = config_path.parent
    model_base_value = payload.get("model_base_path", ".")
    model_base = _expand_path(model_base_value, config_directory)
    checkpoint_value = checkpoint_override or payload.get("checkpoint_path")
    if checkpoint_value is None:
        raise ValueError("Config requires checkpoint_path")
    checkpoint_path = _expand_path(checkpoint_value, model_base)

    return SiglipDinoConfig(
        config_path=config_path,
        dino_model_id=_resolve_model_reference(model["dino_model_id"], model_base),
        dino_hub_model=str(model.get("dino_hub_model", "dinov3_vitl16")),
        dino_hub_repo=str(model.get("dino_hub_repo", DEFAULT_DINOV3_HUB_REPO)),
        siglip_model_id=_resolve_model_reference(model["siglip_model_id"], model_base),
        checkpoint_path=checkpoint_path,
        dino_dim=int(model.get("dino_dim", 1024)),
        siglip_dim=int(model.get("siglip_dim", 1024)),
        projector_hidden_dim=int(model.get("projector_hidden_dim", 2048)),
        projector_num_layers=int(model.get("projector_num_layers", 3)),
        projector_dropout=float(model.get("projector_dropout", 0.0)),
        dino_image_size=int(preprocessing.get("dino_image_size", 224)),
        siglip_image_size=int(preprocessing.get("siglip_image_size", 256)),
        dino_mean=_three_floats(
            preprocessing.get("dino_mean", IMAGENET_MEAN), "dino_mean"
        ),
        dino_std=_three_floats(preprocessing.get("dino_std", IMAGENET_STD), "dino_std"),
        siglip_mean=_three_floats(
            preprocessing.get("siglip_mean", SIGLIP2_MEAN), "siglip_mean"
        ),
        siglip_std=_three_floats(
            preprocessing.get("siglip_std", SIGLIP2_STD), "siglip_std"
        ),
    )


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened = {}
    for key, child in value.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(child, dotted))
    return flattened


def _assert_contains(expected: dict, actual: dict, context: str) -> None:
    actual_flat = _flatten(actual)
    differences = []
    for key, expected_value in sorted(_flatten(expected).items()):
        actual_value = actual_flat.get(key, "<missing>")
        if actual_value != expected_value:
            differences.append(
                f"{key}: expected {expected_value!r}, got {actual_value!r}"
            )
    if differences:
        raise ValueError(f"Incompatible {context}: {'; '.join(differences)}")


def validate_checkpoint_provenance(
    config: SiglipDinoConfig,
    provenance: dict,
    *,
    dino_sha256: str | None,
) -> None:
    dino = {
        "architecture": {
            "feature_dim": config.dino_dim,
            "register_tokens": 4,
            "positional_encoding": "rope",
        }
    }
    if dino_sha256 is not None:
        dino.update(
            {
                "backend": "meta",
                "checkpoint_sha256": dino_sha256,
                "hub_model": config.dino_hub_model,
                "hub_repo": config.dino_hub_repo,
            }
        )
    expected = {
        "features": {
            "dino": dino,
            "preprocessing": {
                "dino_image_size": config.dino_image_size,
                "siglip_image_size": config.siglip_image_size,
                "resize": "square_bilinear_antialias",
            },
        },
        "projector": {
            "input_dim": config.dino_dim,
            "hidden_dim": config.projector_hidden_dim,
            "output_dim": config.siglip_dim,
            "num_layers": config.projector_num_layers,
            "dropout": config.projector_dropout,
        },
    }
    _assert_contains(expected, provenance, "checkpoint provenance")


def _read_checkpoint_provenance(path: Path) -> tuple[dict, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    encoded = metadata.get("provenance")
    if encoded is None:
        raise ValueError("Checkpoint has no provenance metadata")
    provenance = json.loads(encoded)
    if not isinstance(provenance, dict):
        raise ValueError("Checkpoint provenance must be a JSON object")
    actual_digest = hashlib.sha256(_canonical_json_bytes(provenance)).hexdigest()
    recorded_digest = metadata.get("provenance_digest")
    if recorded_digest != actual_digest:
        raise ValueError("Checkpoint provenance digest does not match its payload")
    return provenance, actual_digest


def _fingerprint_weight_reference(reference: str) -> dict[str, object]:
    path = Path(reference).expanduser()
    if path.is_file():
        return {"reference": str(path.resolve()), "sha256": _sha256_file(path)}
    if not path.is_dir():
        return {"reference": reference}
    weight_files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in {".bin", ".pt", ".pth", ".safetensors"}
    )
    digest = hashlib.sha256()
    for weight_file in weight_files:
        digest.update(weight_file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(weight_file).encode("ascii"))
        digest.update(b"\0")
    return {
        "reference": str(path.resolve()),
        "weight_sha256": digest.hexdigest(),
        "weight_file_count": len(weight_files),
    }


def _resolve_amp_dtype(device, precision: str):
    device_type = str(device).split(":", 1)[0]
    if precision == "fp32":
        return None
    if precision == "auto":
        return torch.float16 if device_type == "cuda" else None
    if precision == "fp16":
        return torch.float16 if device_type == "cuda" else None
    if precision == "bf16":
        return torch.bfloat16 if device_type in {"cuda", "cpu"} else None
    raise ValueError(f"Unsupported precision mode: {precision}")


def caption_guided_embeddings(
    b_bar: torch.Tensor,
    c_bar: torch.Tensor,
    d_bar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use one generic caption as content guidance, never as style text."""

    content_reference = functional.normalize(c_bar + d_bar, dim=-1)
    similarity = (b_bar * content_reference).sum(dim=-1, keepdim=True)
    alpha = torch.clamp(1.0 - similarity, min=0.0)
    style = functional.normalize(b_bar - alpha * similarity * content_reference, dim=-1)
    return style, content_reference


class SiglipDinoBackend:
    name = "siglip-dinov3"
    supported_text_modes = frozenset({"image-only", "caption-guided"})

    def __init__(
        self,
        model,
        dino_transform,
        siglip_transform,
        *,
        mode: TextMode = "image-only",
        device: str | torch.device = "cpu",
        precision: str = "auto",
        fingerprint: str,
        preprocessing_fingerprint: str = "unknown",
    ):
        self.mode = validate_backend_mode(self.name, self.supported_text_modes, mode)
        self.model = model
        self.dino_transform = dino_transform
        self.siglip_transform = siglip_transform
        self.device = device
        self.amp_dtype = _resolve_amp_dtype(device, precision)
        self.fingerprint = fingerprint
        self.preprocessing_fingerprint = preprocessing_fingerprint

    @classmethod
    def from_config(
        cls,
        path: Path,
        *,
        checkpoint_override: Path | None = None,
        mode: TextMode = "image-only",
        device: str | torch.device | None = None,
        precision: str = "auto",
    ) -> SiglipDinoBackend:
        config = load_siglip_dino_config(path, checkpoint_override)
        if not config.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Alignment checkpoint not found: {config.checkpoint_path}"
            )
        dino_path = Path(config.dino_model_id)
        dino_sha256 = _sha256_file(dino_path) if dino_path.is_file() else None
        provenance, provenance_digest = _read_checkpoint_provenance(
            config.checkpoint_path
        )
        validate_checkpoint_provenance(config, provenance, dino_sha256=dino_sha256)

        dino = FrozenDINOv3(
            config.dino_model_id,
            hub_model=config.dino_hub_model,
            hub_repo=config.dino_hub_repo,
            expected_dim=config.dino_dim,
        )
        recorded_architecture = provenance["features"]["dino"]["architecture"]
        _assert_contains(
            recorded_architecture,
            dino.provenance["architecture"],
            "runtime DINOv3 architecture",
        )
        siglip = FrozenSigLIP2(config.siglip_model_id)
        projector = AlignmentProjector(
            input_dim=config.dino_dim,
            hidden_dim=config.projector_hidden_dim,
            output_dim=config.siglip_dim,
            num_layers=config.projector_num_layers,
            dropout=config.projector_dropout,
        )
        tensors = load_safetensors(config.checkpoint_path, device="cpu")
        projector_state = {
            key.removeprefix("projector."): value
            for key, value in tensors.items()
            if key.startswith("projector.")
        }
        projector.load_state_dict(projector_state, strict=True)
        model = StyleDecoupler(dino, siglip, projector)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        preprocessing = config.preprocessing_payload()
        preprocessing_fingerprint = hashlib.sha256(
            _canonical_json_bytes(preprocessing)
        ).hexdigest()
        fingerprint_payload = {
            "schema_version": BACKEND_SCHEMA_VERSION,
            "dino": dino.provenance,
            "siglip": _fingerprint_weight_reference(config.siglip_model_id),
            "checkpoint_sha256": _sha256_file(config.checkpoint_path),
            "checkpoint_provenance_digest": provenance_digest,
            "projector": {
                "input_dim": config.dino_dim,
                "hidden_dim": config.projector_hidden_dim,
                "output_dim": config.siglip_dim,
                "num_layers": config.projector_num_layers,
                "dropout": config.projector_dropout,
            },
            "preprocessing": preprocessing,
            "precision": precision,
        }
        fingerprint = hashlib.sha256(
            _canonical_json_bytes(fingerprint_payload)
        ).hexdigest()
        return cls(
            model,
            build_image_transform(
                config.dino_image_size,
                list(config.dino_mean),
                list(config.dino_std),
            ),
            build_image_transform(
                config.siglip_image_size,
                list(config.siglip_mean),
                list(config.siglip_std),
            ),
            mode=mode,
            device=device,
            precision=precision,
            fingerprint=fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )

    def _stack_images(self, images, transform) -> torch.Tensor:
        tensors = []
        for image in images:
            if isinstance(image, Image.Image):
                prepared = image.convert("RGB")
            else:
                with Image.open(image) as source:
                    prepared = source.convert("RGB")
            tensors.append(transform(prepared))
        return torch.stack(tensors).to(self.device)

    @staticmethod
    def _validated_captions(captions, expected_rows: int) -> list[str]:
        values = list(captions or [])
        if len(values) < expected_rows:
            values.extend([None] * (expected_rows - len(values)))
        counts = {"valid": 0, "missing": 0, "empty": 0, "unreadable": 0}
        normalized = []
        for value in values[:expected_rows]:
            if value is None:
                counts["missing"] += 1
            elif not isinstance(value, str):
                counts["unreadable"] += 1
            elif not value.strip():
                counts["empty"] += 1
            else:
                counts["valid"] += 1
                normalized.append(value.strip())
        if counts["valid"] != expected_rows or len(values) != expected_rows:
            details = ", ".join(f"{key}={value}" for key, value in counts.items())
            raise ValueError(
                "Caption-guided mode requires one valid caption per image: " + details
            )
        return normalized

    def encode(self, images, captions=None) -> EmbeddingBatch:
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        else:
            images = list(images)
        if not images:
            raise ValueError("Cannot encode an empty image batch")
        valid_captions = None
        if self.mode == "caption-guided":
            valid_captions = self._validated_captions(captions, len(images))
        dino_pixels = self._stack_images(images, self.dino_transform)
        siglip_pixels = self._stack_images(images, self.siglip_transform)
        device_type = str(self.device).split(":", 1)[0]
        autocast = (
            nullcontext()
            if self.amp_dtype is None
            else torch.autocast(device_type=device_type, dtype=self.amp_dtype)
        )
        with torch.no_grad(), autocast:
            b_bar = functional.normalize(
                self.model.encode_image_siglip(siglip_pixels), dim=-1
            )
            c_bar = functional.normalize(
                self.model.encode_image_dino(dino_pixels), dim=-1
            )
            if self.mode == "caption-guided":
                tokens = self.model.siglip.tokenizer(
                    valid_captions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(self.device)
                attention_mask = tokens["attention_mask"].to(self.device)
                d_bar = functional.normalize(
                    self.model.encode_text(input_ids, attention_mask), dim=-1
                )
                style, content = caption_guided_embeddings(b_bar, c_bar, d_bar)
            else:
                style, content = b_bar, c_bar
        result = EmbeddingBatch(
            style_embeddings=style.detach().float().cpu().numpy(),
            content_embeddings=content.detach().float().cpu().numpy(),
            mode=self.mode,
            backend=self.name,
            model_fingerprint=self.fingerprint,
        )
        result.validate(expected_rows=len(images))
        return result
