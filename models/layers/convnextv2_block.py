import torch
from torch import nn
from torchvision.ops import stochastic_depth

from .global_response_normalization import GlobalResponseNormalization


class ConvNeXtV2Block(nn.Module):
    def __init__(self, channels: int, expansion_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()

        self.channels = channels
        self.expansion_ratio = expansion_ratio
        self.drop_path = drop_path

        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.pwconv1 = nn.Linear(channels, channels * expansion_ratio)
        self.activation = nn.GELU()
        self.grn = GlobalResponseNormalization(channels * expansion_ratio)
        self.pwconv2 = nn.Linear(channels * expansion_ratio, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.activation(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return residual + stochastic_depth(
            x, p=self.drop_path, mode="row", training=self.training
        )
