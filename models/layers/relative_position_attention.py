import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional, Tuple


class RelativePositionAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_rel_pos = use_rel_pos

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("input_size must be provided when use_rel_pos=True")
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, self.head_dim)
            )
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, self.head_dim)
            )

    def _get_rel_pos(
        self, q_size: int, k_size: int, rel_pos: torch.Tensor
    ) -> torch.Tensor:
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        if rel_pos.shape[0] != max_rel_dist:
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        q_coords = torch.arange(q_size, dtype=torch.float32, device=rel_pos.device)[
            :, None
        ]
        q_coords = q_coords * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size, dtype=torch.float32, device=rel_pos.device)[
            None, :
        ]
        k_coords = k_coords * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
            q_size / k_size, 1.0
        )

        return rel_pos_resized[relative_coords.long()]

    def _get_decomposed_rel_pos_bias(
        self,
        q: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        q_h, q_w = q_size
        k_h, k_w = k_size
        rel_h = self._get_rel_pos(q_h, k_h, rel_pos_h)
        rel_w = self._get_rel_pos(q_w, k_w, rel_pos_w)

        bsz, _, dim = q.shape
        q = q.reshape(bsz, q_h, q_w, dim)
        # compute decomposed relative position bias without einsum
        # rel_h: (q_h, k_h, dim) -> (1, q_h, dim, k_h) for batched matmul
        rel_h_t = rel_h.permute(0, 2, 1).unsqueeze(0)
        # q: (bsz, q_h, q_w, dim)
        bias_h = torch.matmul(q, rel_h_t)  # (bsz, q_h, q_w, k_h)
        # rel_w: (q_w, k_w, dim) -> (1, q_w, dim, k_w)
        rel_w_t = rel_w.permute(0, 2, 1).unsqueeze(0)
        # matmul over width dimension: transpose q to (bsz, q_w, q_h, dim)
        bias_w = torch.matmul(q.permute(0, 2, 1, 3), rel_w_t)  # (bsz, q_w, q_h, k_w)
        bias_w = bias_w.permute(0, 2, 1, 3)  # (bsz, q_h, q_w, k_w)
        return (bias_h[:, :, :, :, None] + bias_w[:, :, :, None, :]).reshape(
            -1, q_h * q_w, k_h * k_w
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, height, width, channels = x.shape
        token_count = height * width

        x = x.reshape(bsz, token_count, channels)
        qkv = self.qkv(x).reshape(bsz, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(
            3, bsz * self.num_heads, token_count, self.head_dim
        ).unbind(0)

        q_scaled = q * self.scale
        attn = q_scaled @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = attn + self._get_decomposed_rel_pos_bias(
                q,
                self.rel_pos_h,
                self.rel_pos_w,
                (height, width),
                (height, width),
            )

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = (
            x.view(bsz, self.num_heads, token_count, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, token_count, channels)
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.reshape(bsz, height, width, channels)
