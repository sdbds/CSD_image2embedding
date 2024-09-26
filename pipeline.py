import torch
from transformers import Pipeline
from typing import Union, List
from PIL import Image

class CSDCLIPPipeline(Pipeline):
    def __init__(self, model, processor, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model=model, tokenizer=None, device=device)
        self.processor = processor

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, images):
        if isinstance(images, (str, Image.Image)):
            images = [images]
        
        processed = self.processor(images=images, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(self.device) for k, v in processed.items()}

    def _forward(self, model_inputs):
        pixel_values = model_inputs['pixel_values'].to(self.model.dtype)
        with torch.no_grad():
            features, content_output, style_output = self.model(pixel_values)
        return {"features": features, "content_output": content_output, "style_output": style_output}

    def postprocess(self, model_outputs):
        return {
            "features": model_outputs["features"].cpu().numpy(),
            "content_output": model_outputs["content_output"].cpu().numpy(),
            "style_output": model_outputs["style_output"].cpu().numpy()
        }

    def __call__(self, images: Union[str, List[str], Image.Image, List[Image.Image]]):
        return super().__call__(images)
