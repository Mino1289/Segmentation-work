import torch
from torch import nn

from .patch_embedding import PatchEmbedding
from .transformer_encoder_block import TransformerEncoderBlock


class SamViTConfig:
    def __init__(self, name: str = "huge"):
        config = {
            "base": {
                "patch_size": 16,
                "layers": 12,
                "hidden_size": 768,
                "mlp_size": 3072,
                "num_heads": 12,
                "window_size": 0,
                "global_attn_indexes": (),
            },
            "large": {
                "patch_size": 16,
                "layers": 24,
                "hidden_size": 1024,
                "mlp_size": 4096,
                "num_heads": 16,
                "window_size": 0,
                "global_attn_indexes": (),
            },
            "huge": {
                "patch_size": 14,
                "layers": 32,
                "hidden_size": 1280,
                "mlp_size": 5120,
                "num_heads": 16,
                "window_size": 14,
                "global_attn_indexes": (7, 15, 23, 31),
            },
        }
        if name not in config:
            raise ValueError(
                f"Invalid model size: {name}. Choose from {list(config.keys())}."
            )
        self.name = name
        self.patch_size = config[name]["patch_size"]
        self.layers = config[name]["layers"]
        self.hidden_size = config[name]["hidden_size"]
        self.mlp_size = config[name]["mlp_size"]
        self.num_heads = config[name]["num_heads"]
        self.window_size = config[name]["window_size"]
        self.global_attn_indexes = config[name]["global_attn_indexes"]


class SamViT(nn.Module):
    """
    SAM ViT Backbone (ImageEncoderViT / ViTDet).
    [B, H, W, C] without head by default.
    """

    def __init__(
        self,
        img_size: int = 1024,
        in_channels: int = 3,
        config: SamViTConfig = SamViTConfig(),
        cls_head: bool = False,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            config.patch_size, in_channels, config.hidden_size
        )
        grid_size = img_size // config.patch_size
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    config.hidden_size,
                    config.num_heads,
                    mlp_ratio=config.mlp_size // config.hidden_size,
                    use_rel_pos=True,
                    input_size=(grid_size, grid_size),
                    window_size=0
                    if i in config.global_attn_indexes
                    else config.window_size,
                )
                for i in range(config.layers)
            ]
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid_size, grid_size, config.hidden_size)
        )
        self.norm = nn.Identity()
        self.cls_head = cls_head
        if self.cls_head:
            self.head = nn.Linear(config.hidden_size, num_classes)
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        if self.cls_head:
            return self.head(self.norm(x).mean(dim=(1, 2)))
        return self.head(x)
