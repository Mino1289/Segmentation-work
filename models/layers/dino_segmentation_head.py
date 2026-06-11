import torch
from torch import nn
import torch.nn.functional as F


class DinoSegmentationHead(nn.Module):
    def __init__(
        self, in_channels: int = 384, hidden_channels: int = 256, num_classes: int = 150
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # sortie : [B, num_classes, 37, 37]

        # retrouver la taille native de l'image
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


class DinoLinearSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=150):
        super().__init__()

        self.linear_layer = nn.Conv2d(
            in_channels=in_channels, out_channels=num_classes, kernel_size=1
        )

    def forward(self, concatenated_features: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Args:
            concatenated_features: Tenseur [B, In_Channels, H_patch, W_patch]
            target_size: Tuple (H_orig, W_orig) pour l'image de sortie (ex: 512, 512)
        """
        # 1. Projection linéaire pixel par pixel (ou patch par patch)
        logits = self.linear_layer(
            concatenated_features
        )  # [B, Num_Classes, H_patch, W_patch]

        # 2. Upsampling bilinéaire pour retrouver la taille de l'image d'origine
        logits = F.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )

        return logits
