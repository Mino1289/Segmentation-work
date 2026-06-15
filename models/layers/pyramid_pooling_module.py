import torch
from torch import nn
from torch.nn import functional as F

from ..util import AdaptiveAvgPool2dSafe


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels

        self.pool_proj1 = nn.Sequential(
            AdaptiveAvgPool2dSafe(1),
            nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.pool_proj2 = nn.Sequential(
            AdaptiveAvgPool2dSafe(2),
            nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.pool_proj3 = nn.Sequential(
            AdaptiveAvgPool2dSafe(3),
            nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.pool_proj6 = nn.Sequential(
            AdaptiveAvgPool2dSafe(6),
            nn.Conv2d(self.in_channels, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                self.in_channels + 4 * self.channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is C5 feature map
        pooled1 = self.pool_proj1(x)
        pooled2 = self.pool_proj2(x)
        pooled3 = self.pool_proj3(x)
        pooled6 = self.pool_proj6(x)

        pooled1 = F.interpolate(
            pooled1, size=x.size()[2:], mode="bilinear", align_corners=False
        )
        pooled2 = F.interpolate(
            pooled2, size=x.size()[2:], mode="bilinear", align_corners=False
        )
        pooled3 = F.interpolate(
            pooled3, size=x.size()[2:], mode="bilinear", align_corners=False
        )
        pooled6 = F.interpolate(
            pooled6, size=x.size()[2:], mode="bilinear", align_corners=False
        )

        # Concatenate with original C5 feature map
        out = torch.cat([x, pooled1, pooled2, pooled3, pooled6], dim=1)

        out = self.conv(out)

        return out
