import torch
from typing import Union, List
from PIL import Image
from precision_utils import autocast_context


class CSDCLIPPipeline:
    def __init__(self, model, processor, device=None, amp_dtype=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.processor = processor
        self.device = device
        self.amp_dtype = amp_dtype

    def preprocess(self, images):
        if isinstance(images, (str, Image.Image)):
            images = [images]

        processed = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in processed.items()}

    def _forward(self, model_inputs):
        pixel_values = model_inputs["pixel_values"]
        if self.amp_dtype is None:
            pixel_values = pixel_values.to(self.model.dtype)
        with torch.no_grad():
            with autocast_context(self.device, self.amp_dtype):
                features, content_output, style_output = self.model(pixel_values)
        return {
            "features": features,
            "content_output": content_output,
            "style_output": style_output,
        }

    def postprocess(self, model_outputs):
        return {
            "features": model_outputs["features"].cpu().numpy(),
            "content_output": model_outputs["content_output"].cpu().numpy(),
            "style_output": model_outputs["style_output"].cpu().numpy(),
        }

    def __call__(self, images: Union[str, List[str], Image.Image, List[Image.Image]]):
        model_inputs = self.preprocess(images)
        model_outputs = self._forward(model_inputs)
        return self.postprocess(model_outputs)
