import torch
from torch import nn
from typing import Tuple

from .mask_attention import MaskAttention


class TwoWayTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        skip_first_layer_pe: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.skip_first_layer_pe = skip_first_layer_pe

        # Self-attention on the token embeddings
        self.self_attn = MaskAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            internal_dim=self.embed_dim,
        )
        self.norm1 = nn.LayerNorm(self.embed_dim)

        # Cross-attention
        self.cross_attn_tok_to_img = MaskAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, internal_dim=128
        )
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # Internal MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.embed_dim),
        )
        self.norm3 = nn.LayerNorm(self.embed_dim)

        # Cross-attention
        self.cross_attn_img_to_toks = MaskAttention(
            embed_dim=self.embed_dim, num_heads=self.num_heads, internal_dim=128
        )
        self.norm4 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # step 1: self-attention on tokens
        if self.skip_first_layer_pe:
            q1 = queries.permute(1, 0, 2)
            out, _ = self.self_attn(q1, q1, q1)
        else:
            q1 = torch.add(queries, query_pe).permute(1, 0, 2)
            v1 = queries.permute(1, 0, 2)
            out, _ = self.self_attn(q1, q1, v1)
        tokens_s1 = self.norm1(out.permute(1, 0, 2) + queries)

        # step 2: cross-attention tokens -> image
        q2 = torch.add(tokens_s1, query_pe).permute(1, 0, 2)
        k2 = torch.add(keys, key_pe).permute(1, 0, 2)
        v2 = keys.permute(1, 0, 2)

        out, _ = self.cross_attn_tok_to_img(q2, k2, v2)
        tokens_s2 = self.norm2(out.permute(1, 0, 2) + tokens_s1)

        # step 3: MLP on tokens
        mlp_out = self.mlp(tokens_s2)
        tokens_s3 = self.norm3(mlp_out + tokens_s2)

        # step 4: cross-attention image -> tokens
        q4 = torch.add(keys, key_pe).permute(1, 0, 2)
        k4 = torch.add(tokens_s3, query_pe).permute(1, 0, 2)
        v4 = tokens_s3.permute(1, 0, 2)

        out, _ = self.cross_attn_img_to_toks(q4, k4, v4)
        image_embedding = self.norm4(out.permute(1, 0, 2) + keys)

        return tokens_s3, image_embedding
