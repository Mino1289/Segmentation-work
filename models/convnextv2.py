import torch
from torch import nn
from typing import Tuple

from .layers.convnextv2_block import ConvNeXtV2Block
from .layers.downsampling_block import DownsamplingBlock
from .layers.layer_norm2d import LayerNorm2d

from .util import get_device


class ConvNeXtV2Config:
    def __init__(self, name: str = "base"):
        config = {
            "atto": {
                "channels": [40, 80, 160, 320],
                "blocks": [2, 2, 6, 2],
                "expansion": 4,
            },
            "femto": {
                "channels": [48, 96, 192, 384],
                "blocks": [2, 2, 6, 2],
                "expansion": 4,
            },
            "pico": {
                "channels": [64, 128, 256, 512],
                "blocks": [2, 2, 6, 2],
                "expansion": 4,
            },
            "nano": {
                "channels": [80, 160, 320, 640],
                "blocks": [2, 2, 8, 2],
                "expansion": 4,
            },
            "tiny": {
                "channels": [96, 192, 384, 768],
                "blocks": [3, 3, 9, 3],
                "expansion": 4,
            },
            "small": {
                "channels": [96, 192, 384, 768],
                "blocks": [3, 3, 27, 3],
                "expansion": 4,
            },
            "base": {
                "channels": [128, 256, 512, 1024],
                "blocks": [3, 3, 27, 3],
                "expansion": 4,
            },
            "large": {
                "channels": [192, 384, 768, 1536],
                "blocks": [3, 3, 27, 3],
                "expansion": 4,
            },
            "huge": {
                "channels": [352, 704, 1408, 2816],
                "blocks": [3, 3, 27, 3],
                "expansion": 4,
            },
        }
        if name not in config:
            raise ValueError(
                f"Invalid model size: {name}. Choose from {list(config.keys())}."
            )
        self.name = name
        self.channels = config[name]["channels"]
        self.blocks = config[name]["blocks"]
        self.expansion = config[name]["expansion"]


class ConvNeXtV2(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        config: ConvNeXtV2Config = ConvNeXtV2Config(name="base"),
        features_only: bool = False,
        cls_head: bool = True,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.config = config
        self.features_only = features_only
        self.cls_head = cls_head
        self.num_classes = num_classes

        self.blocks = self.config.blocks
        self.channels = self.config.channels
        self.expansion = self.config.expansion

        self.stem = nn.Sequential(
            nn.Conv2d(input_shape[0], self.channels[0], kernel_size=4, stride=4),
            LayerNorm2d(self.channels[0], eps=1e-6),
        )

        self.stages = nn.ModuleList()

        for stage_idx in range(len(self.channels)):
            stage = nn.Sequential()

            # Si ce n'est pas le premier stage, on commence par un downsample
            if stage_idx > 0:
                stage.add_module(
                    f"downsample{stage_idx}",
                    DownsamplingBlock(
                        self.channels[stage_idx - 1], self.channels[stage_idx]
                    ),
                )

            # On ajoute les blocs de ce stage
            for block_idx in range(self.blocks[stage_idx]):
                stage.add_module(
                    f"block{block_idx}",
                    ConvNeXtV2Block(self.channels[stage_idx], self.expansion),
                )

            self.stages.append(stage)

        if self.cls_head:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.LayerNorm(self.channels[-1], eps=1e-6),
                nn.Linear(self.channels[-1], self.num_classes),
            )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)

        features = []
        for stage in self.stages:
            x = stage(x)
            if self.features_only:
                features.append(x)

        if self.features_only:
            return features

        if self.cls_head:
            x = self.head(x)
        return x

    def load_from_timm(self, timm_model: nn.Module):
        state_dict_timm = timm_model.state_dict()
        state_dict_custom = self.state_dict()

        new_state_dict = {}
        for k_timm, v in state_dict_timm.items():
            k_custom = k_timm

            if k_timm.startswith("stages."):
                parts = k_timm.split(".")
                stage_idx = int(parts[1])

                if parts[2] == "blocks":
                    block_idx = int(parts[3])
                    prefix = f"stages.{stage_idx}.block{block_idx}"

                    if "conv_dw" in k_timm:
                        k_custom = k_timm.replace(
                            f"stages.{stage_idx}.blocks.{block_idx}.conv_dw",
                            f"{prefix}.dwconv",
                        )
                    elif "norm" in k_timm:
                        k_custom = k_timm.replace(
                            f"stages.{stage_idx}.blocks.{block_idx}.norm",
                            f"{prefix}.norm",
                        )
                    else:
                        suffix = ".".join(parts[4:])
                        if suffix.startswith("mlp.fc1"):
                            suffix = suffix.replace("mlp.fc1", "pwconv1")
                        elif suffix.startswith("mlp.fc2"):
                            suffix = suffix.replace("mlp.fc2", "pwconv2")
                        elif suffix == "mlp.grn.weight":
                            suffix = "grn.gamma"
                        elif suffix == "mlp.grn.bias":
                            suffix = "grn.beta"

                        k_custom = f"{prefix}.{suffix}"

                elif parts[2] == "downsample":
                    k_custom = k_timm.replace(
                        f"stages.{stage_idx}.downsample",
                        f"stages.{stage_idx}.downsample{stage_idx}.downsample",
                    )

            elif k_timm == "head.norm.weight" or k_timm == "norm_pre.weight":
                k_custom = "head.2.weight"
            elif k_timm == "head.norm.bias" or k_timm == "norm_pre.bias":
                k_custom = "head.2.bias"
            elif k_timm == "head.fc.weight":
                k_custom = "head.3.weight"
            elif k_timm == "head.fc.bias":
                k_custom = "head.3.bias"

            if k_custom in state_dict_custom:
                new_state_dict[k_custom] = v

        self.load_state_dict(new_state_dict, strict=False)
        return self


if __name__ == "__main__":
    import timm

    device = get_device()

    size = "base"  # Change to "atto", "femto", "pico", "nano", "tiny", "small", "base", "large", or "huge" as needed

    timm_model = timm.create_model(f"convnextv2_{size}", pretrained=True)
    # Remove the classification head for feature extraction
    timm_model.head = nn.Identity()
    timm_model.to(device)

    config = ConvNeXtV2Config(name=size)
    mymodel = ConvNeXtV2(input_shape=(3, 224, 224), config=config, cls_head=False)
    mymodel.load_from_timm(timm_model)
    mymodel.to(device)

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        output = mymodel(dummy_input)

        output_timm = timm_model(dummy_input)
    print("MyModel Output shape:", output.shape)
    print("Timm Model Output shape:", output_timm.shape)
    print("Max absolute difference:", (output - output_timm).abs().max().item())
