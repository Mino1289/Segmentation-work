import torch
from torch import nn
from typing import Tuple

from .two_way_transformer_layer import TwoWayTransformerLayer


class MaskDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (64, 64),
        num_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.num_heads = num_heads

        # 4 discrete mask tokens for the scale: small, medium, large + IoU
        self.iou_token = nn.Embedding(1, self.embed_dim)
        self.num_mask_tokens = 4
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.embed_dim)

        # Puis un petit MLP pour la prédiction de l'IoU :
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_mask_tokens),
        )

        self.num_layers = 2
        self.layers = nn.ModuleList(
            [
                TwoWayTransformerLayer(
                    embed_dim=self.embed_dim, num_heads=self.num_heads
                )
                for _ in range(self.num_layers)
            ]
        )

        self.final_attn = nn.MultiheadAttention(
            self.embed_dim, self.num_heads, dropout=0.1
        )
        self.norm_final = nn.LayerNorm(self.embed_dim)

        self.upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim, self.embed_dim // 4, kernel_size=2, stride=2
            ),  # (B, 256, 64, 64) -> (B, 64, 128, 128)
            nn.GroupNorm(self.embed_dim // 4, self.embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.embed_dim // 4, self.embed_dim // 8, kernel_size=2, stride=2
            ),  # (B, 64, 128, 128) -> (B, 32, 256, 256)
            nn.GroupNorm(self.embed_dim // 8, self.embed_dim // 8),
            nn.GELU(),
        )
        self.output_hypernetworks = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(
                self.embed_dim, self.embed_dim // 8
            ),  # Final projection to match upscaling output (256 -> 32)
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        sparse_prompt_pe: torch.Tensor,
    ) -> torch.Tensor:
        """
        image_embeddings: shape [B, 256, 64, 64]),
        image_pe: shape[B, 256, 64, 64],
        sparse_prompt_embeddings: shape [B, N, 256].
        sparse_prompt_pe: shape [B, N, 256].
        """
        bs = image_embeddings.shape[0]

        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )  # [4+1, 256]
        output_tokens = output_tokens.unsqueeze(0).expand(bs, -1, -1)  # [B, 4+1, 256]
        tokens = torch.cat(
            [output_tokens, sparse_prompt_embeddings], dim=1
        )  # [B, 4+1+N, 256]

        mask_pe = torch.zeros(
            (bs, self.num_mask_tokens, self.embed_dim), device=tokens.device
        )
        query_pe = torch.cat([mask_pe, sparse_prompt_pe], dim=1)  # [B, 4+N, 256]

        image_embeddings = image_embeddings.flatten(2).permute(
            0, 2, 1
        )  # [B, 4096, 256]
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # [B, 4096, 256]

        for layer in self.layers:
            tokens, image_embeddings = layer(
                queries=tokens,
                keys=image_embeddings,
                query_pe=query_pe,
                key_pe=image_pe,
            )

        q = torch.add(tokens, query_pe).permute(1, 0, 2)
        k = torch.add(image_embeddings, image_pe).permute(1, 0, 2)
        v = image_embeddings.permute(1, 0, 2)

        attn_out, _ = self.final_attn(q, k, v)
        tokens = tokens + attn_out.permute(1, 0, 2)
        tokens = self.norm_final(tokens)

        image_embeddings = image_embeddings.permute(0, 2, 1).reshape(
            bs,
            self.embed_dim,
            self.image_embedding_size[0],
            self.image_embedding_size[1],
        )  # [B, 256, 64, 64]

        upscaled = self.upscaling(image_embeddings)  # [B, 32, 256, 256]
        mask_tokens_out = tokens[:, : self.num_mask_tokens, :]  # [B, 4, 256]

        hypernetworks = self.output_hypernetworks(mask_tokens_out)  # [B, 4, 32]

        feat_map = upscaled.flatten(2).permute(0, 2, 1)
        kernels = hypernetworks.permute(0, 2, 1)

        masks_flattened = feat_map @ kernels
        masks_flattened = masks_flattened.permute(0, 2, 1)  # [B, 4, 65536]

        masks = masks_flattened.reshape(bs, self.num_mask_tokens, 256, 256)

        return masks  # [B, 4, 256, 256]
