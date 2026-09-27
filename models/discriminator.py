import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator for high-resolution satellite imagery (2.5m NAIP targets).
    Discriminates local image patches as real or fake.
    """
    def __init__(self, in_channels=4, base_features=64):
        super(PatchGANDiscriminator, self).__init__()
        
        def conv_block(in_f, out_f, stride=2, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=stride, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(in_channels, base_features, stride=2, normalize=False),   # Patch size down x2
            *conv_block(base_features, base_features * 2, stride=2, normalize=True), # Down x4
            *conv_block(base_features * 2, base_features * 4, stride=2, normalize=True), # Down x8
            *conv_block(base_features * 4, base_features * 8, stride=1, normalize=True),
            nn.Conv2d(base_features * 8, 1, kernel_size=4, stride=1, padding=1)  # 1-channel patch decision matrix
        )

    def forward(self, x):
        return self.model(x)
