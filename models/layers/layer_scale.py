import torch
from torch import nn


class LayerScale(nn.Module):
    def __init__(self, channels: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x
