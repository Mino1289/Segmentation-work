import torch
from torch import nn
from typing import Optional, Tuple

from .layer_norm2d import LayerNorm2d
from .position_embedding import PositionEmbedding


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        input_image_size: Tuple[int, int] = (1024, 1024),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size

        # PositionEmbedding layer (128 sin/cos pairs = dim 256)
        self.pe_layer = PositionEmbedding(num_pos_feats=embed_dim // 2)

        # Grouping of discrete embeddings:
        # Index 0: Negative point (bg) | 1: Positive point (fg) | 2: Top-left corner | 3: Bottom-right corner
        self.point_embeddings = nn.Embedding(4, embed_dim)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        # Mask processing (Dense prompts)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=2, stride=2),
            LayerNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 16, kernel_size=2, stride=2),
            LayerNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, embed_dim, kernel_size=1),
        )

    def forward(
        self,
        points: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # (coords [B, N, 2], labels [B, N])
        boxes: Optional[torch.Tensor] = None,  # [B, N, 4]
        masks: Optional[torch.Tensor] = None,  # [B, 1, 256, 256]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dynamically determine the active batch size
        bs = (
            points[0].shape[0]
            if points is not None
            else (
                boxes.shape[0]
                if boxes is not None
                else (masks.shape[0] if masks is not None else 1)
            )
        )
        device = self.point_embeddings.weight.device

        # Initialization of the empty sequence to store sparse tokens (Sparse)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=device)

        # 1. Management of POINTS
        if points is not None:
            coords, labels = points
            coords = coords + 0.5  # Shift to the pixel center as Meta does

            # Calculation of base PE
            point_embeddings = self.pe_layer.forward_with_coords(
                coords, self.input_image_size
            )

            # Application of learned identifiers (Foreground vs Background vs Ignored/Padding)
            point_embeddings[labels == -1] += self.not_a_point_embed.weight
            point_embeddings[labels == 0] += self.point_embeddings.weight[0]
            point_embeddings[labels == 1] += self.point_embeddings.weight[1]

            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        # 2. Management of BOXES
        if boxes is not None:
            boxes = boxes + 0.5
            box_coords = boxes.reshape(
                -1, 2, 2
            )  # Separate into [B*N, 2 (TopLeft/BottomRight), 2 (X/Y)]

            box_embeddings = self.pe_layer.forward_with_coords(
                box_coords, self.input_image_size
            )
            box_embeddings[:, 0, :] += self.point_embeddings.weight[2]  # Badge Top-Left
            box_embeddings[:, 1, :] += self.point_embeddings.weight[
                3
            ]  # Badge Bottom-Right

            # Return to sequential format [B, N*2, 256]
            box_embeddings = box_embeddings.reshape(bs, -1, self.embed_dim)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # 3. Management of MASKS (Dense)
        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            # If no mask, duplication of the default embedding over the entire grid [B, 256, 64, 64]
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings
