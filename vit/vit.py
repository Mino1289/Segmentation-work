import torch
from torch import nn

from patch_embedding import PatchEmbedding
from transformer_encoder_block import TransformerEncoderBlock


class ViTConfig:
    def __init__(
        self,
        name: str = "base",
    ):
        config = {
            "base": {
                "patch_size": 16,
                "layers": 12,
                "hidden_size": 768,
                "mlp_size": 3072,
                "num_heads": 12,
            },
            "large": {
                "patch_size": 16,
                "layers": 24,
                "hidden_size": 1024,
                "mlp_size": 4096,
                "num_heads": 16,
            },
            "huge": {
                "patch_size": 14,
                "layers": 32,
                "hidden_size": 1280,
                "mlp_size": 5120,
                "num_heads": 16,
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


class ViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        config: ViTConfig = ViTConfig(),
        cls_head: bool = True,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, config.patch_size, in_channels, config.hidden_size
        )
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    config.hidden_size,
                    config.num_heads,
                    mlp_ratio=config.mlp_size // config.hidden_size,
                )
                for _ in range(config.layers)
            ]
        )
        grid_size = img_size // config.patch_size  # 1024 // 16 = 64
        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid_size, grid_size, config.hidden_size)
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        self.cls_head = cls_head
        if self.cls_head:
            self.head = nn.Linear(config.hidden_size, num_classes)
        else:
            self.head = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed

        B, H, W, C = x.shape
        x = x.view(B, H * W, C)

        x = self.encoder(x)
        x = self.norm(x)

        x = x.view(B, H, W, C)
        if self.cls_head:
            return self.head(x[:, 0])
        return self.head(x)


if __name__ == "__main__":
    model_config = ViTConfig(name="huge")
    model_config.patch_size = 16

    model = ViT(
        img_size=1024,
        in_channels=3,
        config=model_config,
        cls_head=False,
    )
    model.to("xpu")

    x = torch.randn(1, 3, 1024, 1024).to("xpu")
    with torch.no_grad():
        out = model(x)

    print(out.shape)
