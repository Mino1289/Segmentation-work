import torch
from torch import nn
import timm
from typing import List

from .convnextv2 import ConvNeXtV2, ConvNeXtV2Config
from .layers.feature_pyramid_network import FeaturePyramidNetwork
from .layers.object_segmentation_head import ObjectSegmentationHead


class Backbone(nn.Module):
    def __init__(
        self,
        model_size: str = "base",
        input_shape: tuple = (3, 512, 512),
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_size = model_size
        self.pretrained = pretrained
        self.input_shape = input_shape

        self.config = ConvNeXtV2Config(name=self.model_size)

        self.backbone = ConvNeXtV2(
            input_shape=self.input_shape,
            config=self.config,
            features_only=True,
            cls_head=False,
        )
        if self.pretrained:
            timm_model = timm.create_model(
                f"convnextv2_{self.model_size}", pretrained=True
            )
            self.backbone.load_from_timm(timm_model)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.backbone(x)
        return features  # C2, C3, C4, C5 (Ck => 1/2^k of input resolution)


class UPerNet(nn.Module):
    def __init__(self, backbone: nn.Module, channels: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.backbone_channels = self.backbone.config.channels
        self.channels = channels
        self.num_classes = num_classes

        self.fpn = FeaturePyramidNetwork(
            backbone_channels=self.backbone_channels, out_channels=self.channels
        )
        self.head = ObjectSegmentationHead(
            channels=self.channels, num_classes=self.num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features = self.backbone(x)
        # fpn_features = self.fpn(features)
        # out = self.head(fpn_features)  # fpn_features est un tuple (P2, P3, P4, P5)
        features = self.backbone(x)
        def decode_step(*feats):
            fpn_feats = self.fpn(feats)
            return self.head(fpn_feats)
        out = decode_step(*features)
        return out


if __name__ == "__main__":
    # test usage
    backbone = Backbone(model_size="base", pretrained=False)
    model = UPerNet(backbone, channels=512, num_classes=150)
    model.eval()
    model.to("xpu")
    input_tensor = torch.randn(1, 3, 512, 512).to("xpu")
    with torch.no_grad():
        output = model(input_tensor)
    print(output.shape)  # Expected output shape: [1, 150, 512, 512]
