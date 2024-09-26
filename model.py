import copy
import torch.nn as nn
import clip
from huggingface_hub import PyTorchModelHubMixin
from transformers import PretrainedConfig

class CSDCLIPConfig(PretrainedConfig):
    model_type = "csd_clip"

    def __init__(
        self,
        name="csd_large",
        embedding_dim=1024,
        feature_dim=1024,
        content_dim=768,
        style_dim=768,
        content_proj_head="default",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.embedding_dim = embedding_dim
        self.content_proj_head = content_proj_head
        self.task_specific_params = None  # Add this line

class CSD_CLIP(nn.Module, PyTorchModelHubMixin):
    """backbone + projection head"""
    def __init__(self, name='vit_large',content_proj_head='default'):
        super(CSD_CLIP, self).__init__()
        self.content_proj_head = content_proj_head
        if name == 'vit_large':
            clipmodel, _ = clip.load("ViT-L/14")
            self.backbone = clipmodel.visual
            self.embedding_dim = 1024
            self.feature_dim = 1024
            self.content_dim = 768
            self.style_dim = 768
            self.name = "csd_large"
        elif name == 'vit_base':
            clipmodel, _ = clip.load("ViT-B/16")
            self.backbone = clipmodel.visual
            self.embedding_dim = 768 
            self.feature_dim = 512
            self.content_dim = 512
            self.style_dim = 512
            self.name = "csd_base"
        else:
            raise Exception('This model is not implemented')

        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        self.last_layer_content = copy.deepcopy(self.backbone.proj)

        self.backbone.proj = None
        
        self.config = CSDCLIPConfig(
            name=self.name,
            embedding_dim=self.embedding_dim,
            feature_dim=self.feature_dim,
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            content_proj_head=self.content_proj_head
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

        style_output = feature @ self.last_layer_style
        style_output = nn.functional.normalize(style_output, dim=1, p=2)

        content_output = feature @ self.last_layer_content
        content_output = nn.functional.normalize(content_output, dim=1, p=2)
        
        return feature, content_output, style_output
