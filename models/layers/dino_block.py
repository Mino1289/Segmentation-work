import torch
from torch import nn

from .dino_MLP import DinoMLP
from .dino_self_attention import DinoSelfAttention
from .layer_scale import LayerScale

class DinoBlock(nn.Module):
    def __init__(self, embed_dim:int, num_heads:int):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.attn = DinoSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ls1 = LayerScale(channels=embed_dim, init_value=1e-5)
        
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.mlp = DinoMLP(embed_dim)
        self.ls2 = LayerScale(channels=embed_dim, init_value=1e-5)
        
    def forward(self, x:torch.Tensor):
        #additionner le x pour les skip connections
        #tout en 1 ligne pour garder toutes les infos
        x = x + self.ls1(self.attn(self.norm1(x)))
        
        x = x + self.ls2(self.mlp(self.norm2(x)))
        
        return x