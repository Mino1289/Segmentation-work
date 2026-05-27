import torch
from torch import nn
from torch.nn import functional as F

from typing import Optional, Tuple
from .relative_position_attention import RelativePositionAttention

# This module implements a transformer encoder block with optional relative positional attention and windowed attention.


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        emb_size: int = 768,
        heads: int = 12,
        dropout: float = 0.0,
        mlp_ratio: int = 4,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
        window_size: int = 0,
    ):
        super().__init__()

        self.window_size = window_size
        self.norm1 = nn.LayerNorm(emb_size, eps=1e-6)
        self.attn = RelativePositionAttention(
            emb_size,
            num_heads=heads,
            qkv_bias=True,
            attn_drop=dropout,
            proj_drop=dropout,
            use_rel_pos=use_rel_pos,
            input_size=(window_size, window_size) if window_size > 0 else input_size,
        )
        self.norm2 = nn.LayerNorm(emb_size, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(emb_size * mlp_ratio, emb_size),
        )
        self.dropout = nn.Dropout(dropout)

    def _window_partition(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, height, width, channels = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        padded_h = height + pad_h
        padded_w = width + pad_w

        x = x.view(
            batch_size,
            padded_h // self.window_size,
            self.window_size,
            padded_w // self.window_size,
            self.window_size,
            channels,
        )
        x = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(-1, self.window_size, self.window_size, channels)
        )
        return x, (padded_h, padded_w)

    def _window_unpartition(
        self,
        windows: torch.Tensor,
        hw: Tuple[int, int],
        pad_hw: Tuple[int, int],
    ) -> torch.Tensor:
        padded_h, padded_w = pad_hw
        height, width = hw
        batch_size = windows.shape[0] // (
            padded_h * padded_w // self.window_size // self.window_size
        )
        x = windows.view(
            batch_size,
            padded_h // self.window_size,
            padded_w // self.window_size,
            self.window_size,
            self.window_size,
            -1,
        )
        x = (
            x.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(batch_size, padded_h, padded_w, -1)
        )
        return x[:, :height, :width, :].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, channels = x.shape

        shortcut = x
        x = self.norm1(x)

        pad_hw = None
        if self.window_size > 0:
            x, pad_hw = self._window_partition(x)

        x = self.attn(x)

        if self.window_size > 0:
            x = self._window_unpartition(x, (height, width), pad_hw)

        x = shortcut + x

        x = x.reshape(batch_size, height * width, channels)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        x = x.reshape(batch_size, height, width, channels)
        return x
