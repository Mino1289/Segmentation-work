import torch
from torch import nn
from torch.nn import functional as F


class ObjectSegmentationHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.in_channels = self.channels * 4  # (2048)

        # Fuse
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        P2, P3, P4, P5 = features

        # Interpolate P3, P4, P5 to P2's resolution
        target_size = P2.size()[2:]
        P3 = F.interpolate(P3, size=target_size, mode="bilinear", align_corners=False)
        P4 = F.interpolate(P4, size=target_size, mode="bilinear", align_corners=False)
        P5 = F.interpolate(P5, size=target_size, mode="bilinear", align_corners=False)

        out = torch.cat([P2, P3, P4, P5], dim=1)  # channels wise cat (512 * 4 = 2048)

        out = self.fusion_conv(out)
        out = self.classifier(out)

        # Final interpolation to original image size (assuming input was 4x P2's resolution)
        out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=False)
        return out
