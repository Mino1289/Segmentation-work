import torch
from torch import nn

import os

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import LayerNorm2d
from vit.vit import ViT, ViTConfig


class ImageEncoder(nn.Module):
    def __init__(self, model_size: str = "huge"):
        super().__init__()
        self.vit = ViT(
            img_size=1024,
            in_channels=3,
            # SAM uses the "huge" model, but with 16 heads instead of 14.
            config=ViTConfig(name=model_size).patch_size(16),
            cls_head=False,
        )

        self.convs = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            LayerNorm2d(256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.vit(x).reshape(x.shape[0], 1024 // 16, 1024 // 16, -1)
        emb = emb.permute(0, 3, 1, 2)  # (B, C, H', W')
        out = self.convs(emb)
        return out
