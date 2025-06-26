import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfConditionedModulation(nn.Module):
    """
    Self-conditioned normalization and modulation block:
    - Learns per-channel mean/variance from content features
    - Applies modulation (normed * var + mean)
    - Blends with stylized features using external attention map
    """
    def __init__(self, in_channels, return_delta=False, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or in_channels // 2

        # Content encoder: Conv -> IN -> GAP -> Conv1x1
        self.content_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels, affine=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU()
        )

        # Projection MLP
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )

        # Heads for modulation stats
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

        self.var_head = nn.Sequential(
            nn.Linear(hidden_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

        self.return_delta = return_delta

    def forward(self, content, original):
        B, C, H, W = content.shape

        # Compute base mean and std from content
        base_mean = content.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        base_std = content.std(dim=[2, 3], keepdim=True) + 1e-6

        original_mean = original.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        original_std = original.std(dim=[2, 3], keepdim=True) + 1e-6

        # Encode and project for modulation deltas
        encoded = self.content_encoder(content).view(B, -1)
        proj = self.projector(encoded)

        d_mean = self.mean_head(proj).view(B, C, 1, 1)
        d_var = self.var_head(proj).view(B, C, 1, 1)

        # Apply deltas on base stats
        modulated_mean = base_mean + d_mean
        modulated_var = base_std + F.softplus(d_var)  # ensure positivity

        # Normalize and apply affine modulation
        normed = (original - original_mean)/original_std # self.norm(content)
        out = normed * modulated_var + modulated_mean

        if self.return_delta:
          out = (out,d_mean,d_var)

        return out
