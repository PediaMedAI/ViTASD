import torch
import torch.nn as nn

from timm.models import create_model
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import PatchEmbed

from lib.sngp import Laplace


attn_setting_dict = {
    'tiny': {
        'dim': 192,
        'num_heads': 3
    },
    'small': {
        'dim': 384,
        'num_heads': 6
    },
    'base': {
        'dim': 768,
        'num_heads': 12,
    },
    'large': {
        'dim': 1024,
        'num_heads': 16,
    }
}



class ViTASD_SNGP(nn.Module):
    def __init__(self, backbone: str, num_classes, num_data, batch_size, variant, drop_rate, drop_path_rate, input_size):
        super(ViTASD_SNGP, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        self.backbone: VisionTransformer = create_model(
            backbone,
            pretrained=True,
            num_classes=attn_setting_dict[variant]['dim'],
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            drop_block_rate=None,
            img_size=input_size
        )
        self.embed_dim = self.backbone.embed_dim

        num_deep_features = attn_setting_dict[variant]['dim']
        num_gp_features = 0 # 128
        normalize_gp_features = True
        num_random_features = 1024
        mean_field_factor = 25
        ridge_penalty = 1
        lengthscale = 2

        self.model = Laplace(
            self.backbone,
            num_deep_features,
            num_gp_features,
            normalize_gp_features,
            num_random_features,
            num_classes,
            num_data,
            batch_size,
            mean_field_factor,
            ridge_penalty,
            lengthscale
        )


    def forward(self, x):
        x = self.model(x)
        return x



