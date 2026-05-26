import torch
from torch import nn
import numpy as np
from typing import Tuple


class PositionEmbedding(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    Mainly from official SAM Implementation:
    """

    def __init__(self, num_pos_feats: int = 128, scale: float = 1.0):
        super().__init__()

        self.register_buffer("gauss_matrix", scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Central calculus of Fourier Features on normalized coordinates [0,1]"""
        coords = 2 * coords - 1  # Go in [-1, 1]
        coords = coords @ self.gauss_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # [..., 256]

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate the complete dense PE grid for the Image Embedding (e.g., 64x64)"""
        h, w = size
        device = self.gauss_matrix.device

        # Creation of the pixel-center normalized coordinate grid
        y_embed = (torch.arange(h, device=device) + 0.5) / h
        x_embed = (torch.arange(w, device=device) + 0.5) / w
        y, x = torch.meshgrid(y_embed, x_embed, indexing="ij")

        grid = torch.stack([x, y], dim=-1)  # [H, W, 2]
        return self._pe_encoding(grid).permute(2, 0, 1)  # [256, H, W]

    def forward_with_coords(
        self, coords: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Encode points or bounding box corners (e.g., on a 1024x1024 scale)"""
        coords_norm = coords.clone().float()
        coords_norm[..., 0] /= image_size[1]  # Normalization X
        coords_norm[..., 1] /= image_size[0]  # Normalization Y
        return self._pe_encoding(coords_norm)
