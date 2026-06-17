import torch
from torch import nn
from typing import Tuple

from layers.layer_norm2d import LayerNorm2d
from layers.downsampling_block import DownsamplingBlock
from layers.convnext_block import ConvNeXtBlock

from util import AdaptiveAvgPool2dSafe, get_device


class ConvNeXtConfig:
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
            "xlarge": {
                "channels": [256, 512, 1024, 2048],
                "blocks": [3, 3, 27, 3],
                "expansion": 4,
            },
            "xxlarge": {
                "channels": [384, 768, 1536, 3072],
                "blocks": [3, 4, 30, 3],
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


class ConvNeXt(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        config: ConvNeXtConfig = ConvNeXtConfig(name="base"),
        cls_head: bool = True,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.config = config
        self.cls_head = cls_head
        self.num_classes = num_classes

        self.blocks = self.config.blocks
        self.channels = self.config.channels
        self.expansion = self.config.expansion

        self.model = nn.Sequential()

        stem = nn.Sequential(
            nn.Conv2d(input_shape[0], self.channels[0], kernel_size=4, stride=4),
            LayerNorm2d(self.channels[0], eps=1e-6),
        )
        self.model.add_module("stem", stem)

        for stage_idx in range(len(self.channels)):
            for block_idx in range(self.blocks[stage_idx]):
                self.model.add_module(
                    f"stage{stage_idx}_block{block_idx}",
                    ConvNeXtBlock(self.channels[stage_idx], self.expansion),
                )
            if stage_idx < len(self.channels) - 1:
                self.model.add_module(
                    f"downsample{stage_idx}",
                    DownsamplingBlock(
                        self.channels[stage_idx], self.channels[stage_idx + 1]
                    ),
                )

        if self.cls_head:
            self.model.add_module("global_avg_pool", AdaptiveAvgPool2dSafe((1, 1)))
            self.model.add_module("flatten", nn.Flatten(1))
            self.model.add_module(
                "layer_norm", nn.LayerNorm(self.channels[-1], eps=1e-6)
            )
            self.model.add_module(
                "classifier", nn.Linear(self.channels[-1], self.num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def load_from_timm(self, timm_model: nn.Module):
        state_dict_timm = timm_model.state_dict()
        state_dict_custom = self.state_dict()

        new_state_dict = {}
        for k_timm, v in state_dict_timm.items():
            k_custom = k_timm

            if k_timm.startswith("stem."):
                k_custom = f"model.{k_timm}"
            elif k_timm.startswith("stages."):
                parts = k_timm.split(".")
                stage_idx = int(parts[1])

                if parts[2] == "blocks":
                    block_idx = int(parts[3])
                    prefix = f"model.stage{stage_idx}_block{block_idx}"

                    suffix = ".".join(parts[4:])
                    if suffix.startswith("mlp.fc1"):
                        suffix = suffix.replace("mlp.fc1", "fc1")
                    elif suffix.startswith("mlp.fc2"):
                        suffix = suffix.replace("mlp.fc2", "fc2")
                    elif suffix == "gamma":
                        suffix = "layer_scale.gamma"

                    k_custom = f"{prefix}.{suffix}"

                elif parts[2] == "downsample":
                    prefix = f"model.downsample{stage_idx - 1}"
                    k_custom = k_timm.replace(f"stages.{stage_idx}.", f"{prefix}.")

            elif k_timm == "head.norm.weight" or k_timm == "norm_pre.weight":
                k_custom = "model.layer_norm.weight"
            elif k_timm == "head.norm.bias" or k_timm == "norm_pre.bias":
                k_custom = "model.layer_norm.bias"
            elif k_timm == "head.fc.weight":
                k_custom = "model.classifier.weight"
            elif k_timm == "head.fc.bias":
                k_custom = "model.classifier.bias"

            if k_custom in state_dict_custom:
                new_state_dict[k_custom] = v

        self.load_state_dict(new_state_dict, strict=False)
        return self


if __name__ == "__main__":
    import timm

    device = get_device()

    size = "base"  # Change to "small", "base", "large", or "xlarge" as needed

    timm_model = timm.create_model(f"convnext_{size}", pretrained=False)
    # Remove the classification head for feature extraction
    timm_model.head = nn.Identity()
    timm_model.to(device)

    config = ConvNeXtConfig(name=size)
    mymodel = ConvNeXt(input_shape=(3, 224, 224), config=config, cls_head=False)
    mymodel.load_from_timm(timm_model)
    mymodel.to(device)

    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        output = mymodel(dummy_input)

        output_timm = timm_model(dummy_input)
    print("MyModel Output shape:", output.shape)
    print("Timm Model Output shape:", output_timm.shape)
    print("Max absolute difference:", (output - output_timm).abs().max().item())
