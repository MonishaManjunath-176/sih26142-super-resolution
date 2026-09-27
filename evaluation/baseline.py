import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import SEN2NAIPDataset
from evaluation.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_sam,
)


# ============================================================
# CONFIGURATION
# ============================================================

LR_DIR = "data/sen2naipv2_extracted/test/lr"
HR_DIR = "data/sen2naipv2_extracted/test/hr"

SCALE = 4
BATCH_SIZE = 1


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SIH26142 — BICUBIC BASELINE EVALUATION")
print("=" * 60)

print(f"Device: {device}")
print(f"Test LR directory: {LR_DIR}")
print(f"Test HR directory: {HR_DIR}")
print()


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not os.path.exists(LR_DIR):
    raise FileNotFoundError(
        f"Test LR directory not found: {LR_DIR}"
    )

if not os.path.exists(HR_DIR):
    raise FileNotFoundError(
        f"Test HR directory not found: {HR_DIR}"
    )


# ============================================================
# DATASET
# ============================================================

test_dataset = SEN2NAIPDataset(
    lr_dir=LR_DIR,
    hr_dir=HR_DIR,
    in_channels=4,
    transform=None,
    is_train=False,
    scale=SCALE,
)

print(f"Test samples: {len(test_dataset)}")
print()


# ============================================================
# DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

print("Test DataLoader created.")
print()


# ============================================================
# EVALUATION
# ============================================================

total_psnr = 0.0
total_ssim = 0.0
total_sam = 0.0

num_samples = 0

print("=" * 60)
print("EVALUATING BICUBIC BASELINE")
print("=" * 60)


with torch.no_grad():

    for batch_index, batch in enumerate(
        test_loader,
        start=1
    ):

        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        # ----------------------------------------------------
        # BICUBIC UPSAMPLING
        # ----------------------------------------------------

        sr = F.interpolate(
            lr,
            size=hr.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        sr_cpu = sr.squeeze(0).cpu()
        hr_cpu = hr.squeeze(0).cpu()

        psnr = calculate_psnr(
            sr_cpu,
            hr_cpu
        )

        ssim = calculate_ssim(
            sr_cpu,
            hr_cpu
        )

        sam = calculate_sam(
            sr_cpu,
            hr_cpu
        )

        total_psnr += float(psnr)
        total_ssim += float(ssim)
        total_sam += float(sam)

        num_samples += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            batch_index <= 5
            or batch_index % 100 == 0
            or batch_index == len(test_loader)
        ):

            print(
                f"[{batch_index:4d}/{len(test_loader)}] "
                f"PSNR: {psnr:.4f} | "
                f"SSIM: {ssim:.4f} | "
                f"SAM: {sam:.4f}°"
            )


# ============================================================
# FINAL RESULTS
# ============================================================

avg_psnr = total_psnr / num_samples
avg_ssim = total_ssim / num_samples
avg_sam = total_sam / num_samples


print()
print("=" * 60)
print("FINAL BICUBIC BASELINE RESULTS")
print("=" * 60)

print(f"Test samples : {num_samples}")
print(f"PSNR         : {avg_psnr:.4f} dB")
print(f"SSIM         : {avg_ssim:.4f}")
print(f"SAM          : {avg_sam:.4f}°")

print("=" * 60)
print("BICUBIC EVALUATION COMPLETE")
print("=" * 60)