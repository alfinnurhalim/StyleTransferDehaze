import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfConditionedModulation(nn.Module):
    def __init__(self, in_channels, return_delta=False, hidden_channels=None, style_dim=None, detach_original=True):
        super().__init__()
        hidden_channels = hidden_channels or in_channels // 2
        style_dim = style_dim or hidden_channels

        self.return_delta = return_delta
        self.detach_original = detach_original

        # === Content Feature Encoder ===
        self.content_encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_channels, affine=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU()
        )

        self.content_projector = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )

        # === Style Feature Projector ===
        self.style_projector = nn.Sequential(
            nn.Linear(style_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )

        # === Mixing Module ===
        mix_dim = 2 * hidden_channels
        self.mix_block = nn.Sequential(
            nn.Linear(mix_dim, mix_dim),
            nn.ReLU(),
            nn.Linear(mix_dim, mix_dim),
            nn.ReLU()
        )

        # === Output Modulation Heads ===
        self.mean_head = nn.Linear(mix_dim, in_channels)
        self.var_head = nn.Linear(mix_dim, in_channels)

    def forward(self, content, style_code):
        B, C, H, W = content.shape
        original = content.clone().detach() if self.detach_original else content

        # === Stats from original (pre-modulation) ===
        mean = original.mean(dim=[2, 3], keepdim=True)
        std = original.std(dim=[2, 3], keepdim=True) + 1e-6

        # === Encode content ===
        content_encoded = self.content_encoder(content).view(B, -1)
        content_proj = self.content_projector(content_encoded)

        # === Encode style code ===
        style_proj = self.style_projector(style_code)

        # === Mix ===
        mix = torch.cat([content_proj, style_proj], dim=1)
        mixed = self.mix_block(mix)

        # === Residual AdaIN outputs ===
        d_mean = self.mean_head(mixed).view(B, C, 1, 1)
        d_var = self.var_head(mixed).view(B, C, 1, 1)

        mod_mean = mean + d_mean
        mod_std = std + F.softplus(d_var)

        # === Normalize and Modulate ===
        normed = (original - mean) / std
        out = normed * mod_std + mod_mean

        if self.return_delta:
            return out, d_mean, d_var
        return out
