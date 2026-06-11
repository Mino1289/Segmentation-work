import torch
from torch import nn


class DinoMLP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return out
