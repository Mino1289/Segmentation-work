import torch
from torch import nn

from .layer_scale import LayerScale


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion_ratio: int = 4):
        super().__init__()

        self.channels = channels
        self.expansion_ratio = expansion_ratio

        self.conv_dw = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.fc1 = nn.Linear(channels, channels * expansion_ratio)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(channels * expansion_ratio, channels)
        self.layer_scale = LayerScale(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.layer_scale(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return residual + x
