import torch
from torch import nn
import torch.nn.functional as F


class DinoSegmentationHead(nn.Module):
    def __init__(self, in_channels: int=384, hidden_channels :int=256, num_classes: int=150 ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        
    def forward(self, x, target_size:tuple):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        #sortie : [B, num_classes, 37, 37]
        
        #retrouver la taille native de l'image
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x
        
        
