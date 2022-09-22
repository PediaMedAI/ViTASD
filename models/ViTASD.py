import torch
import torch.nn as nn

from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed


class ViTASD(nn.Module):
    def __init__(self, backbone: str, num_classes, drop_rate, drop_path_rate, input_size,
                 input_size_2):
        super(ViTASD, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.backbone: VisionTransformer = create_model(
            backbone,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=None,
            img_size=input_size
        )
        self.embed_dim = self.backbone.embed_dim


    def forward(self, x):
        x = self.backbone.patch_embed(x)
        x = x + self.backbone.pos_embed
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:, 0]
        x = self.backbone.head(x1)
        return x
