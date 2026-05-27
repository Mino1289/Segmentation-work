import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple


class MaskAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, internal_dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, internal_dim)
        self.k_proj = nn.Linear(embed_dim, internal_dim)
        self.v_proj = nn.Linear(embed_dim, internal_dim)
        self.out_proj = nn.Linear(internal_dim, embed_dim)
        self.num_heads = num_heads
        self.internal_dim = internal_dim
        self.head_dim = internal_dim // num_heads

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input shape: [seq_len, batch_size, embed_dim]
        # Linear projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        b = q.shape[1]
        q_len = q.shape[0]
        kv_len = k.shape[0]

        q = q.reshape(q_len, b, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.reshape(kv_len, b, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.reshape(kv_len, b, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # [batch_size, num_heads, q_len, head_dim]
        out = out.permute(2, 0, 1, 3).reshape(q_len, b, self.internal_dim)

        return self.out_proj(out), attn
