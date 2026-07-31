"""CSD-CLIP model, preprocessing, precision, and embedding backend."""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image
from transformers import PretrainedConfig

from .base import EmbeddingBatch, precision_identity

CSD_IMAGE_SIZE = 336
CSD_BACKEND_SCHEMA_VERSION = 1


class CSDClipConfig(PretrainedConfig):
    model_type = "csd_clip"

    def __init__(
        self,
        name="csd_large",
        embedding_dim=1024,
        feature_dim=1024,
        content_dim=768,
        style_dim=768,
        content_proj_head="default",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.content_dim = content_dim
        self.style_dim = style_dim
        self.content_proj_head = content_proj_head
        self.task_specific_params = None


class CSDClip(nn.Module, PyTorchModelHubMixin):
    """CLIP visual backbone with independent content and style projections."""

    def __init__(self, name="vit_large", content_proj_head="default"):
        super().__init__()
        try:
            import clip
        except ImportError as error:
            raise ImportError(
                "CSD inference requires the OpenAI CLIP package"
            ) from error

        self.content_proj_head = content_proj_head
        if name in {"vit_large", "csd_large"}:
            clip_model, _ = clip.load("ViT-L/14")
            self.embedding_dim = 1024
            self.feature_dim = 1024
            self.content_dim = 768
            self.style_dim = 768
            self.name = "csd_large"
        elif name in {"vit_base", "csd_base"}:
            clip_model, _ = clip.load("ViT-B/16")
            self.embedding_dim = 768
            self.feature_dim = 512
            self.content_dim = 512
            self.style_dim = 512
            self.name = "csd_base"
        else:
            raise ValueError(f"Unsupported CSD backbone: {name}")

        self.backbone = clip_model.visual
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        self.last_layer_content = copy.deepcopy(self.backbone.proj)
        self.backbone.proj = None
        self.config = CSDClipConfig(
            name=self.name,
            embedding_dim=self.embedding_dim,
            feature_dim=self.feature_dim,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            content_proj_head=self.content_proj_head,
        )

    def get_config(self):
        return self.config.to_dict()

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        feature = self.backbone(input_data)
        style_output = nn.functional.normalize(
            feature @ self.last_layer_style, dim=1, p=2
        )
        content_output = nn.functional.normalize(
            feature @ self.last_layer_content, dim=1, p=2
        )
        return feature, content_output, style_output


def get_device_type(device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def resolve_amp_dtype(device, precision: str = "auto"):
    device_type = get_device_type(device)
    precision = precision.lower()
    if precision == "fp32":
        return None
    if precision == "auto":
        return torch.float16 if device_type == "cuda" else None
    if precision == "fp16":
        return torch.float16 if device_type == "cuda" else None
    if precision == "bf16":
        return torch.bfloat16 if device_type in {"cuda", "cpu"} else None
    raise ValueError(f"Unsupported precision mode: {precision}")


def autocast_context(device, amp_dtype):
    device_type = get_device_type(device)
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=amp_dtype)


def normalize_image_batch(images):
    if isinstance(images, (str, Path, Image.Image)):
        return [images], True
    return list(images), False


def stack_transformed_images(images, transform, device):
    transformed = [transform(image.convert("RGB")) for image in images]
    return torch.stack(transformed, dim=0).to(device)


def preprocess_csd_image(image: Image.Image | str | Path) -> Image.Image:
    """Apply the square white padding used by the original CSD workflow."""

    if isinstance(image, Image.Image):
        prepared = image.convert("RGB")
    else:
        with Image.open(image) as source:
            prepared = source.convert("RGB")
    image_array = np.asarray(prepared)
    size = max(image_array.shape[:2])
    pad_x = size - image_array.shape[1]
    pad_y = size - image_array.shape[0]
    pad_left = pad_x // 2
    pad_top = pad_y // 2
    padded = np.pad(
        image_array,
        (
            (pad_top, pad_y - pad_top),
            (pad_left, pad_x - pad_left),
            (0, 0),
        ),
        mode="constant",
        constant_values=255,
    )
    return Image.fromarray(padded).resize(
        (CSD_IMAGE_SIZE, CSD_IMAGE_SIZE), Image.Resampling.LANCZOS
    )


class CSDClipPipeline:
    """Small processor/model adapter retained for direct CSD inference."""

    def __init__(self, model, processor, device=None, amp_dtype=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.processor = processor
        self.device = device
        self.amp_dtype = amp_dtype

    def preprocess(self, images):
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        processed = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {key: value.to(self.device) for key, value in processed.items()}

    def _forward(self, model_inputs):
        pixel_values = model_inputs["pixel_values"]
        if self.amp_dtype is None:
            pixel_values = pixel_values.to(self.model.dtype)
        with torch.no_grad(), autocast_context(self.device, self.amp_dtype):
            features, content_output, style_output = self.model(pixel_values)
        return {
            "features": features,
            "content_output": content_output,
            "style_output": style_output,
        }

    @staticmethod
    def postprocess(model_outputs):
        return {
            key: value.detach().cpu().numpy() for key, value in model_outputs.items()
        }

    def __call__(self, images):
        return self.postprocess(self._forward(self.preprocess(images)))


def _normalize_rows(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("CSD produced an embedding row with a zero norm")
    return np.asarray(array / norms, dtype=np.float32)


class CSDClipBackend:
    """Image-only implementation of the shared embedding backend contract."""

    name = "csd"
    supported_text_modes = frozenset({"image-only"})

    def __init__(
        self,
        model,
        processor,
        *,
        model_name: str,
        processor_name: str,
        device=None,
        precision: str = "auto",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.processor = processor
        self.device = device
        self.precision = precision
        amp_dtype = resolve_amp_dtype(device, precision)
        self.pipeline = CSDClipPipeline(
            model=model,
            processor=processor,
            device=device,
            amp_dtype=amp_dtype,
        )
        preprocessing_payload = {
            "schema_version": 1,
            "processor_name": processor_name,
            "square_padding": "white_centered",
            "pre_resize": {
                "size": CSD_IMAGE_SIZE,
                "interpolation": "lanczos",
            },
        }
        self.preprocessing_fingerprint = hashlib.sha256(
            json.dumps(
                preprocessing_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        fingerprint_payload = {
            "schema_version": CSD_BACKEND_SCHEMA_VERSION,
            "model_name": model_name,
            "processor_name": processor_name,
            "embedding_dim": getattr(model, "embedding_dim", None),
            "content_dim": getattr(model, "content_dim", None),
            "style_dim": getattr(model, "style_dim", None),
            "precision": precision_identity(precision, amp_dtype),
        }
        encoded = json.dumps(
            fingerprint_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.fingerprint = hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        processor_name: str,
        *,
        device=None,
        precision: str = "auto",
    ) -> CSDClipBackend:
        from transformers import CLIPProcessor

        model = CSDClip.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        processor = CLIPProcessor.from_pretrained(processor_name)
        return cls(
            model,
            processor,
            model_name=model_name,
            processor_name=processor_name,
            device=device,
            precision=precision,
        )

    def encode(self, images, captions=None) -> EmbeddingBatch:
        del captions
        image_batch, _ = normalize_image_batch(images)
        if not image_batch:
            raise ValueError("Cannot encode an empty image batch")
        prepared = [preprocess_csd_image(image) for image in image_batch]
        outputs = self.pipeline(prepared)
        result = EmbeddingBatch(
            style_embeddings=_normalize_rows(outputs["style_output"]),
            content_embeddings=_normalize_rows(outputs["content_output"]),
            mode="image-only",
            backend=self.name,
            model_fingerprint=self.fingerprint,
        )
        result.validate(expected_rows=len(prepared))
        return result
