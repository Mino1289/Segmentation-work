import torch
from torch import nn
from torch.nn import functional as F

from typing import List

from .pyramid_pooling_module import PyramidPoolingModule


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, backbone_channels: List[int], out_channels: int):
        super().__init__()
        self.backbone_channels = backbone_channels
        self.out_channels = out_channels

        self.ppm = PyramidPoolingModule(
            in_channels=backbone_channels[-1],
            channels=backbone_channels[-1] // 4,
            out_channels=out_channels,
        )

        self.convC2 = nn.Conv2d(
            self.backbone_channels[0], self.out_channels, kernel_size=1
        )
        self.convC3 = nn.Conv2d(
            self.backbone_channels[1], self.out_channels, kernel_size=1
        )
        self.convC4 = nn.Conv2d(
            self.backbone_channels[2], self.out_channels, kernel_size=1
        )

        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_conv3 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_conv4 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.fpn_conv5 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        C2, C3, C4, C5 = features

        P5 = self.ppm(C5)
        P4 = self.convC4(C4) + F.interpolate(
            P5, size=C4.size()[2:], mode="bilinear", align_corners=False
        )
        P3 = self.convC3(C3) + F.interpolate(
            P4, size=C3.size()[2:], mode="bilinear", align_corners=False
        )
        P2 = self.convC2(C2) + F.interpolate(
            P3, size=C2.size()[2:], mode="bilinear", align_corners=False
        )

        P2 = self.fpn_conv2(P2)
        P3 = self.fpn_conv3(P3)
        P4 = self.fpn_conv4(P4)
        P5 = self.fpn_conv5(P5)

        return P2, P3, P4, P5
