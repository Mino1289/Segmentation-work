import torch
from torch import nn

from models.layers.rotary_embedding import apply_rotary_pos_emb


class DinoSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        # q, k et v dans la même projection dans dino
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        sin: torch.Tensor = None,
        cos: torch.Tensor = None,
        num_prefix_tokens: int = 0,
    ) -> torch.Tensor:
        # x de dimension [B, N, C]
        b, n, c = x.shape
        head_dim = c // self.num_heads

        # qkv_tensor est de dimension [B, N, 3C]
        qkv_tensor = self.qkv(x)

        redimensionnement = qkv_tensor.reshape(b, n, 3, self.num_heads, head_dim)
        redimensionnement = redimensionnement.permute(2, 0, 3, 1, 4).contiguous()
        # redimensonnment de dimension [3, B, num_heads, N+1, head_dim]

        q, k, v = torch.unbind(redimensionnement, dim=0)
        # dimensions de q, k et v :[B, num_heads, N+1, head_dim]

        if sin is not None and cos is not None:
            if num_prefix_tokens > 0:
                q_prefix, q_patches = q[:, :, :num_prefix_tokens, :], q[:, :, num_prefix_tokens:, :]
                k_prefix, k_patches = k[:, :, :num_prefix_tokens, :], k[:, :, num_prefix_tokens:, :]
                q = torch.cat(
                    [q_prefix, apply_rotary_pos_emb(q_patches, sin, cos)], dim=2
                )
                k = torch.cat(
                    [k_prefix, apply_rotary_pos_emb(k_patches, sin, cos)], dim=2
                )
            else:
                q = apply_rotary_pos_emb(q, sin, cos)
                k = apply_rotary_pos_emb(k, sin, cos)

        attention_tensor = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        # dimesion de attention_tensor : [B, num_heads, N+1, head_dim]

        attention_tensor = attention_tensor.permute(0, 2, 1, 3).contiguous()
        attention_tensor = attention_tensor.reshape(b, n, -1)
        # contiguous pour bien remodeler le tenseur et pas juste les pointeurs (permutés)

        return self.proj(attention_tensor)
