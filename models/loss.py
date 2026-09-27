import torch
import torch.nn as nn


def to_reflectance(tensor):
    """Convert the model's Tanh range [-1, 1] to reflectance range [0, 1]."""
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


class SpectralConsistencyLoss(nn.Module):
    """
    Mean spectral-angle loss in radians.

    The loss compares the spectral vector of each predicted pixel
    with the corresponding high-resolution reference pixel.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, sr, hr):
        sr_reflectance = to_reflectance(sr)
        hr_reflectance = to_reflectance(hr)

        dot = torch.sum(sr_reflectance * hr_reflectance, dim=1)

        sr_norm = torch.linalg.vector_norm(
            sr_reflectance,
            dim=1
        )
        hr_norm = torch.linalg.vector_norm(
            hr_reflectance,
            dim=1
        )

        cosine = dot / (sr_norm * hr_norm + self.eps)
        cosine = torch.clamp(
            cosine,
            -1.0 + self.eps,
            1.0 - self.eps
        )

        return torch.acos(cosine).mean()


class SuperResolutionLoss(nn.Module):
    """
    Reconstruction + spectral-consistency objective.

    L1:
        Encourages pixel-level reconstruction fidelity.

    Spectral consistency:
        Encourages the predicted RGBN spectral signature to remain
        close to the high-resolution reference.

    No adversarial/discriminator loss is used in the CNN + Attention
    architecture.
    """

    def __init__(self, lambda_l1=1.0, lambda_spectral=0.1):
        super().__init__()

        self.l1_loss = nn.L1Loss()
        self.spectral_loss = SpectralConsistencyLoss()

        self.lambda_l1 = lambda_l1
        self.lambda_spectral = lambda_spectral

    def forward(self, sr, hr):
        l1 = self.l1_loss(sr, hr)
        spectral = self.spectral_loss(sr, hr)

        total = (
            self.lambda_l1 * l1
            + self.lambda_spectral * spectral
        )

        return total, {
            "l1_loss": l1.item(),
            "spectral_loss": spectral.item(),
        }


# Keep the old class name available so older imports do not immediately break.
GeneratorLoss = SuperResolutionLoss
