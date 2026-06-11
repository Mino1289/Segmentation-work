import torch
from torch import nn
from typing import Tuple

from models.dino_feature_extractor import DinoV3FeatureExtractor
from models.layers.dino_block import DinoBlock
from models.layers.dino_segmentation_head import DinoLinearSegmentationHead
from models.layers.patch_embedding import PatchEmbedding
from models.layers.rotary_embedding import RopePositionEmbedding2D


class DinoV3Config:
    def __init__(self, name: str = "S"):

        configs = {
            "S": {"embed_dim": 384, "num_heads": 6, "depth": 12},
            "B": {"embed_dim": 768, "num_heads": 12, "depth": 18},
            "L": {"embed_dim": 1024, "num_heads": 16, "depth": 24},
        }
        if name not in configs:
            raise ValueError(
                f"Taille '{name}' non reconnue. Choisissez parmi 'S', 'B', 'L'."
            )
        self.name = name
        self.embed_dim = configs[name]["embed_dim"]
        self.num_heads = configs[name]["num_heads"]
        self.depth = configs[name]["depth"]


class DinoV3(nn.Module):
    def __init__(
        self,
        config: DinoV3Config = DinoV3Config(name="S"),
        image_size: int = 512,
        patch_size: int = 16,
    ):
        super().__init__()

        self.config = config

        self.embed_dim = self.config.embed_dim
        self.num_heads = self.config.num_heads
        self.depth = self.config.depth

        self.rope = RopePositionEmbedding2D(
            embed_dim=self.embed_dim, num_heads=self.num_heads
        )

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, emb_size=self.embed_dim
        )

        self.blocks = nn.ModuleList(
            [
                DinoBlock(self.embed_dim, self.num_heads, dinov3=True)
                for _ in range(self.depth)
            ]
        )

        self.layer_norm = nn.LayerNorm(normalized_shape=self.embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Patch embedding -> donne maintenant [B, H_p * W_p, embed_dim]
        x = self.patch_embed(x)
        b, h, w, c = x.shape

        x = x.reshape(B, -1, self.embed_dim)  # [B, N, embed_dim] avec N = H_p * W_p

        # 2. Calcul du nombre de patchs en hauteur et largeur
        hp = H // self.patch_embed.proj.kernel_size[0]
        wp = W // self.patch_embed.proj.kernel_size[1]

        # 3. RoPE 2D -> donne sin, cos de forme [hp * wp, head_dim]
        sin, cos = self.rope(hp, wp, device=x.device)

        # 4. Blocs de transformer avec RoPE
        for block in self.blocks:
            x = block(x, sin=sin, cos=cos)

        # 5. Normalisation finale
        x = self.layer_norm(x)

        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class DinoV3LinearADE20K(nn.Module):
    def __init__(
        self,
        backbone: DinoV3FeatureExtractor,
        num_classes: int = 150,
    ):
        super().__init__()
        self.backbone = backbone
        self.segmentation_head = DinoLinearSegmentationHead(
            in_channels=backbone.embed_dim * 4, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 1. Extraire les features concaténées des 4 derniers blocs
        features = self.backbone(x)  # [B, embed_dim*4, H_patch, W_patch]

        # 2. Passer par la tête de segmentation linéaire
        target_size = (H, W)  # (H_orig, W_orig)
        logits = self.segmentation_head(
            features, target_size=target_size
        )  # [B, num_classes, H_orig, W_orig]

        return logits
