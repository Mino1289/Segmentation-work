import math
import torch
from torch import Tensor, nn


def apply_rotary_pos_emb(
    x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor
) -> torch.Tensor:
    """Applique l'encodage rotatif (RoPE) sur un tenseur de requêtes (Q) ou de clés (K).

    x: [B, num_heads, N, head_dim]
    sin, cos: [N, head_dim] (générés par RopePositionEmbedding2D)
    """
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
    """Implémentation épurée de RoPE 2D (Axial) pour les modèles de vision."""

    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0):
        super().__init__()
        # Chaque tête est divisée en 4 sections (sin/cos pour X et sin/cos pour Y)
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 4 == 0, (
            "La dimension par tête doit être divisible par 4 pour le RoPE 2D."
        )

        # Calcul des périodes (les fréquences de calcul pour RoPE)
        # Équivalent à : base ^ (4 * i / head_dim)
        half_d = self.head_dim // 4
        periods = base ** (torch.arange(half_d, dtype=torch.float32) / half_d)
        self.register_buffer("periods", periods, persistent=True)

    def forward(self, H: int, W: int, device: torch.device) -> tuple[Tensor, Tensor]:
        # 1. Génération de la grille de coordonnées normalisées entre -1 et 1
        grid_h = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
        grid_w = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W

        # meshgrid avec indexing="ij" -> [H, W]
        mesh_h, mesh_w = torch.meshgrid(grid_h, grid_w, indexing="ij")

        # Stack et passage de [0, 1] à [-1, 1]
        coords = torch.stack([mesh_h, mesh_w], dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [H * W, 2]
        coords = 2.0 * coords - 1.0

        # 2. Calcul des angles pour RoPE
        # coords[:, :, None] -> [H*W, 2, 1] | self.periods[None, None, :] -> [1, 1, head_dim//4]
        angles = (
            2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        )  # [H*W, 2, head_dim//4]

        # Fusion des dimensions X et Y pour obtenir l'encodage complet de la tête
        angles = angles.flatten(1, 2)  # [H*W, head_dim//2]

        # Répétition pour correspondre à la taille de la tête complète (sin et cos)
        angles = angles.tile(2)  # [H*W, head_dim]

        # 3. Calcul des fonctions trigonométriques
        cos = torch.cos(angles)  # [H*W, head_dim]
        sin = torch.sin(angles)  # [H*W, head_dim]

        return sin, cos
