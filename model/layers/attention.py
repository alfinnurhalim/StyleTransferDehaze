
import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, mid_channels=64, kernel_size=7):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.attn(x)