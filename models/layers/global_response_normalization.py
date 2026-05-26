import torch
from torch import nn


class GlobalResponseNormalization(nn.Module):
    """Global Response Normalization (GRN) module as described in the ConvNeXtV2 paper."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies GRN to the input tensor.
        x shape is expected to be (B, H, W, C)."""
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x
