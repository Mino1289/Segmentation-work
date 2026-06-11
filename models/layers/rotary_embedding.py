import torch
from torch import nn
import math


def apply_rotary_pos_emb(
    x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    """Applique l'encodage rotatif (RoPE) sur un tenseur de requêtes (Q) ou de clés (K)."""
    half_dim = x.shape[-1] // 2

    # Séparation en deux de la dimension de la tête (head_dim)
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    # Formulation mathématique de la rotation : [-x2, x1]
    x_rotated = torch.cat((-x2, x1), dim=-1)

    # Ajustement des dimensions [N, head_dim] -> [1, 1, N, head_dim] pour le broadcasting
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)

    # (x * cos) + (x_rotated * sin)
    return (x * cos) + (x_rotated * sin)


class RopePositionEmbedding2D(nn.Module):
    """Implémentation épurée du RoPE DINOv3 de timm (sans les augmentations d'entraînement)."""

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        half_d = self.head_dim // 4

        # 1. Calcul des périodes avec base=100.0 en Float32 (comme timm)
        exponents = torch.arange(half_d, dtype=torch.float32) / half_d
        periods = base**exponents
        self.register_buffer("periods", periods, persistent=False)

    def forward(
        self, H: int, W: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 2. Grille normalisée centrée sur 0.5 puis mappée sur [-1, 1]
        grid_h = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
        grid_w = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W

        mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

        # [H*W, 2]
        coords = torch.stack([mesh_h, mesh_w], dim=-1).flatten(0, 1)
        coords = 2.0 * coords - 1.0  # Mapping vers [-1, 1]

        # 3. Calcul des angles spécifiques à DINOv3 (avec 2 * Pi)
        # coords[:, :, None] -> [H*W, 2, 1]
        # self.periods[None, None, :] -> [1, 1, half_d]
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]

        # [H*W, 2 * half_d] = [H*W, head_dim // 2]
        angles = angles.flatten(1, 2)

        # Duplication pour matcher l'application du RoPE -> [H*W, head_dim]
        angles = angles.tile(2)

        return torch.sin(angles), torch.cos(angles)
