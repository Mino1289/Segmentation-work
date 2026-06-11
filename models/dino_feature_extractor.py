import torch
import torch.nn as nn

from .layers.dino_segmentation_head import DinoLinearSegmentationHead
from .dinov2 import DinoV2
from .dinov3 import DinoV3


class BaseDinoFeatureExtractor(nn.Module):
    """Classe de base pour l'extraction de features multi-couches de modèles DINO."""

    def __init__(self, dino_model: nn.Module):
        super().__init__()
        self.dino = dino_model
        self.features = {}
        self.hooks = []

        # Cibler automatiquement les 4 derniers blocs
        total_blocks = len(self.dino.blocks)
        self.target_layers = list(range(total_blocks - 4, total_blocks))

        for idx in self.target_layers:
            layer = self.dino.blocks[idx]
            hook = layer.register_forward_hook(self._get_hook(idx))
            self.hooks.append(hook)

    def _get_hook(self, name: int):
        def hook(model, input, output):
            self.features[name] = output

        return hook

    def _extract_patches(self, feat: torch.Tensor) -> torch.Tensor:
        """Méthode abstraite à surcharger pour nettoyer la séquence de tokens (ex: virer le CLS)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Calculer la géométrie de la grille de patchs à partir de l'input
        hp = x.shape[2] // self.dino.patch_embed.proj.kernel_size[0]
        wp = x.shape[3] // self.dino.patch_embed.proj.kernel_size[1]

        # 2. Forward complet (les hooks interceptent les features des blocs cibles)
        _ = self.dino(x)

        output_features = []
        for idx in self.target_layers:
            feat = self.features[idx]

            # 3. Extraction spécifique à la version de DINO (DINOv2 vs DINOv3)
            patch_feats = self._extract_patches(feat)  # [B, N_patches, embed_dim]

            # 4. Reshape en grille spatiale: [B, N, C] -> [B, C, hp, wp]
            B, _, C = patch_feats.shape
            patch_feats = patch_feats.transpose(1, 2).view(B, C, hp, wp)

            output_features.append(patch_feats)

        # 5. Concaténation finale sur la dimension des channels
        return torch.cat(output_features, dim=1)

    def remove_hooks(self):
        """Nettoie les hooks pour libérer la mémoire."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


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
