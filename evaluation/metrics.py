import math
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def _to_unit_range(tensor):
    """Converts tensor/array from [-1, 1] range to [0, 1] range as float32 numpy array."""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.min() < 0:
            tensor = (tensor + 1.0) / 2.0
        return torch.clamp(tensor, 0.0, 1.0).numpy().astype(np.float32)
    else:
        arr = np.array(tensor, dtype=np.float32)
        if arr.min() < 0:
            arr = (arr + 1.0) / 2.0
        return np.clip(arr, 0.0, 1.0).astype(np.float32)


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between generated SR and ground truth HR image.
    img1, img2: PyTorch Tensors or Numpy arrays (C, H, W)
    """
    arr1 = _to_unit_range(img1)
    arr2 = _to_unit_range(img2)

    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * math.log10(max_val / math.sqrt(mse)))


def calculate_ssim(img1, img2):
    """
    Computes Structural Similarity Index Measure (SSIM).
    img1, img2: PyTorch Tensors or Numpy arrays (C, H, W) in range [0, 1]
    """
    arr1 = _to_unit_range(img1)
    arr2 = _to_unit_range(img2)

    if arr1.ndim == 3 and arr1.shape[0] in [1, 3, 4]:
        arr1 = np.transpose(arr1, (1, 2, 0))
        arr2 = np.transpose(arr2, (1, 2, 0))

    if HAS_SKIMAGE:
        channel_axis = 2 if arr1.ndim == 3 else None
        return float(ssim_fn(arr1, arr2, data_range=1.0, channel_axis=channel_axis))
    else:
        c1 = (0.01 * 1.0) ** 2
        c2 = (0.03 * 1.0) ** 2
        mu1 = np.mean(arr1)
        mu2 = np.mean(arr2)
        var1 = np.var(arr1)
        var2 = np.var(arr2)
        cov = np.cov(arr1.flatten(), arr2.flatten())[0, 1]
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
        return float(ssim_val)


def calculate_sam(img1, img2, eps=1e-7):
    """
    Computes Spectral Angle Mapper (SAM) in degrees to measure spectral consistency across multi-spectral bands.
    img1, img2: Multi-spectral image arrays (C, H, W)
    """
    arr1 = _to_unit_range(img1)
    arr2 = _to_unit_range(img2)

    c = arr1.shape[0]
    v1 = arr1.reshape(c, -1)
    v2 = arr2.reshape(c, -1)

    dot = np.sum(v1 * v2, axis=0)
    norm1 = np.linalg.norm(v1, axis=0)
    norm2 = np.linalg.norm(v2, axis=0)

    cosine = dot / (norm1 * norm2 + eps)
    cosine = np.clip(cosine, -1.0 + eps, 1.0 - eps)
    angle_rad = np.arccos(cosine)
    angle_deg = np.degrees(np.mean(angle_rad))
    return float(angle_deg)


def plot_visual_comparison(lr_img, sr_img, hr_img, save_path=None):
    """
    Generates side-by-side visual comparison between:
    - Sentinel-2 10m LR input
    - Generated 2.5m SR image
    - NAIP 2.5m HR ground truth reference
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def prep_rgb(tensor):
        arr = _to_unit_range(tensor)
        if arr.ndim == 3 and arr.shape[0] >= 3:
            rgb = arr[:3, :, :]
            rgb = np.transpose(rgb, (1, 2, 0))
        elif arr.ndim == 3:
            rgb = arr[0, :, :]
        else:
            rgb = arr
        return np.ascontiguousarray(np.clip(rgb, 0.0, 1.0), dtype=np.float32)

    axes[0].imshow(prep_rgb(lr_img))
    axes[0].set_title("Input: 10m Sentinel-2 (LR)")
    axes[0].axis("off")

    axes[1].imshow(prep_rgb(sr_img))
    axes[1].set_title("Generated: 2.5m Super-Resolution (SR)")
    axes[1].axis("off")

    axes[2].imshow(prep_rgb(hr_img))
    axes[2].set_title("Target: 2.5m NAIP (HR Reference)")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
