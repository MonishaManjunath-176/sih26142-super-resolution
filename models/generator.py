import torch
import torch.nn as nn


class ResidualCNNBlock(nn.Module):
    """Lightweight residual CNN block for local feature extraction."""
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.activation = nn.PReLU(channels)

    def forward(self, x):
        return self.activation(x + self.block(x))


class ChannelAttention(nn.Module):
    """
    CBAM channel attention.
    Learns which feature channels are more important.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_attention = self.mlp(self.avg_pool(x))
        max_attention = self.mlp(self.max_pool(x))

        attention = self.sigmoid(avg_attention + max_attention)
        return x * attention


class SpatialAttention(nn.Module):
    """
    CBAM spatial attention.
    Learns which spatial locations contain useful information.
    """
    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)

        attention_input = torch.cat([avg_map, max_map], dim=1)
        attention = self.sigmoid(self.conv(attention_input))

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module:
    channel attention followed by spatial attention.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction=reduction
        )
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class UpsampleBlock(nn.Module):
    """2x learned upsampling using Conv2d + PixelShuffle."""
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.PReLU(channels)
        )

    def forward(self, x):
        return self.block(x)


class CNNAttentionGenerator(nn.Module):
    """
    CNN + CBAM Attention hybrid network for 4x multispectral
    satellite super-resolution.

    Input:
        (B, 4, H, W)

    Output:
        (B, 4, 4H, 4W)

    The network keeps the existing 4-band RGBN input/output and
    uses two PixelShuffle(2) stages for 4x spatial upscaling.
    """

    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        base_features=64,
        upscale_factor=4,
        num_res_blocks=4,
        attention_reduction=8,
    ):
        super().__init__()

        if upscale_factor != 4:
            raise ValueError(
                "CNNAttentionGenerator currently supports only a 4x upscale factor."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor

        # Initial CNN feature extraction.
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_features,
                kernel_size=3,
                padding=1
            ),
            nn.PReLU(base_features),
        )

        # Deep local feature extraction.
        self.residual_blocks = nn.Sequential(
            *[
                ResidualCNNBlock(base_features)
                for _ in range(num_res_blocks)
            ]
        )

        # Attention refinement.
        self.attention = CBAM(
            base_features,
            reduction=attention_reduction
        )

        # Feature refinement before upscaling.
        self.refine = nn.Sequential(
            nn.Conv2d(
                base_features,
                base_features,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(base_features),
            nn.PReLU(base_features),
        )

        # Two x2 stages = x4 total upscaling.
        self.upscale = nn.Sequential(
            UpsampleBlock(base_features),
            UpsampleBlock(base_features),
        )

        # Final 4-band reconstruction.
        self.output = nn.Conv2d(
            base_features,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        # Shallow features.
        shallow = self.head(x)

        # CNN residual feature learning.
        features = self.residual_blocks(shallow)

        # Attention-enhanced features.
        features = self.attention(features)

        # Residual feature refinement.
        features = self.refine(features) + shallow

        # 4x learned upscaling.
        features = self.upscale(features)

        # Reconstruct 4-band RGBN output in [-1, 1].
        return self.tanh(self.output(features))


# Backward-friendly alias for code that may refer to a generic generator name.
AttentionCNNGenerator = CNNAttentionGenerator
