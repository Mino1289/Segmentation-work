import torch
from torch import nn

from .layer_norm2d import LayerNorm2d
from .sam_vit import SamViT, SamViTConfig


class ImageEncoder(nn.Module):
    def __init__(self, model_size: str = "huge"):
        super().__init__()
        self.config = SamViTConfig(name=model_size)
        self.config.patch_size = 16  # SAM ViT-H utilise patch 16, pas 14

        self.sam_vit = SamViT(
            img_size=1024,
            in_channels=3,
            config=self.config,
            cls_head=False,
        )

        self.convs = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            LayerNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.sam_vit(x).reshape(x.shape[0], 1024 // 16, 1024 // 16, -1)
        emb = emb.permute(0, 3, 1, 2)
        return self.convs(emb)
