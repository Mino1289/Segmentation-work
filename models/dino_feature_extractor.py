import torch
import torch.nn as nn

from .layers.dino_segmentation_head import DinoLinearSegmentationHead
from .dinov2 import DinoV2
from .dinov3 import DinoV3


class BaseDinoFeatureExtractor(nn.Module):
    """Classe de base pour l'extraction de features."""

    def __init__(self, dino_model):
        super().__init__()
        self.dino = dino_model

    def _extract_patches(self, feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hp = x.shape[2] // self.dino.patch_embed.proj.kernel_size[0]
        wp = x.shape[3] // self.dino.patch_embed.proj.kernel_size[1]

        # On appelle le forward modifié (voir section suivante)
        blocks_outputs = self.dino(x, return_all_blocks=True)

        output_features = []
        for feat in blocks_outputs:
            # Appelle la méthode spécifique à la sous-classe (V2 ou V3)
            patch_feats = self._extract_patches(feat)

            B, _, C = patch_feats.shape
            patch_feats = patch_feats.transpose(1, 2).contiguous().view(B, C, hp, wp)
            output_features.append(patch_feats)

        return torch.cat(output_features, dim=1)


class DinoV2FeatureExtractor(BaseDinoFeatureExtractor):
    def __init__(self, dino_model: DinoV2):
        super().__init__(dino_model)

    def _extract_patches(self, feat: torch.Tensor) -> torch.Tensor:
        # Pour DinoV2, on retire le CLS token situé à l'index 0
        return feat[:, 1:, :]


class DinoV3FeatureExtractor(BaseDinoFeatureExtractor):
    def __init__(self, dino_model: DinoV3):
        super().__init__(dino_model)

    def _extract_patches(self, feat: torch.Tensor) -> torch.Tensor:
        return feat[:, self.dino.num_storage_tokens :, :]


class DinoLinearADE20K(nn.Module):
    def __init__(
        self,
        backbone: BaseDinoFeatureExtractor,
        num_classes: int = 150,
    ):
        super().__init__()
        self.backbone = backbone
        self.segmentation_head = DinoLinearSegmentationHead(
            in_channels=backbone.dino.embed_dim * 4, num_classes=num_classes
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
