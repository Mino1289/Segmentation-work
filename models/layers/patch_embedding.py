import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = 16, in_channels: int = 3, emb_size: int = 768):
        super().__init__()
        # An easier Conv2d
        self.proj = nn.Conv2d(
            in_channels, emb_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 1)  # Sortie : [B, emb_size, H_patch, W_patch]
        return x
