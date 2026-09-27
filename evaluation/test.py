import os
import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import SEN2NAIPDataset
from data.transforms import get_transforms
from models.generator import CNNAttentionGenerator
from evaluation.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_sam,
)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = "configs/config.yaml"

LR_DIR = "data/sen2naipv2_extracted/test/lr"
HR_DIR = "data/sen2naipv2_extracted/test/hr"

CHECKPOINT_PATH = "outputs/checkpoints/best_generator.pth"

IN_CHANNELS = 4
OUT_CHANNELS = 4
UPSCALE_FACTOR = 4

BASE_FEATURES = 64
NUM_RES_BLOCKS = 4
ATTENTION_REDUCTION = 8

BATCH_SIZE = 1


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SIH26142 — TEST SET EVALUATION")
print("=" * 60)

print(f"Device: {device}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Test LR directory: {LR_DIR}")
print(f"Test HR directory: {HR_DIR}")
print()


# ============================================================
# CHECK FILES / DIRECTORIES
# ============================================================

if not os.path.exists(LR_DIR):
    raise FileNotFoundError(
        f"Test LR directory not found: {LR_DIR}"
    )

if not os.path.exists(HR_DIR):
    raise FileNotFoundError(
        f"Test HR directory not found: {HR_DIR}"
    )

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = SEN2NAIPDataset(
    lr_dir=LR_DIR,
    hr_dir=HR_DIR,
    in_channels=IN_CHANNELS,
    transform=get_transforms(
        is_train=False,
        scale=UPSCALE_FACTOR
    ),
    is_train=False,
    scale=UPSCALE_FACTOR,
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
# MODEL
# ============================================================

model = CNNAttentionGenerator(
    in_channels=IN_CHANNELS,
    out_channels=OUT_CHANNELS,
    base_features=BASE_FEATURES,
    upscale_factor=UPSCALE_FACTOR,
    num_res_blocks=NUM_RES_BLOCKS,
    attention_reduction=ATTENTION_REDUCTION,
)

model = model.to(device)


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

# Handle either:
# 1. raw state_dict
# 2. checkpoint dictionary containing generator_state_dict
# 3. checkpoint dictionary containing state_dict

if isinstance(checkpoint, dict):

    if "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    else:
        # Could itself already be a state_dict
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True,
)

model.eval()

print("Checkpoint loaded successfully.")
print()


# ============================================================
# TEST EVALUATION
# ============================================================

total_psnr = 0.0
total_ssim = 0.0
total_sam = 0.0

num_samples = 0


print("=" * 60)
print("EVALUATING TEST SET")
print("=" * 60)


with torch.no_grad():

    for batch_index, batch in enumerate(test_loader, start=1):

        lr = batch["lr"].to(
            device,
            non_blocking=True
        )

        hr = batch["hr"].to(
            device,
            non_blocking=True
        )

        sample_ids = batch["sample_id"]

        # ----------------------------------------------------
        # SUPER-RESOLUTION
        # ----------------------------------------------------

        sr = model(lr)

        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        # Metrics are calculated sample-by-sample
        # because batch size = 1.

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
                f"ID: {sample_ids[0]} | "
                f"PSNR: {psnr:.4f} | "
                f"SSIM: {ssim:.4f} | "
                f"SAM: {sam:.4f}°"
            )


# ============================================================
# FINAL AVERAGES
# ============================================================

avg_psnr = total_psnr / num_samples
avg_ssim = total_ssim / num_samples
avg_sam = total_sam / num_samples


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("FINAL TEST SET RESULTS")
print("=" * 60)

print(f"Test samples : {num_samples}")
print(f"PSNR         : {avg_psnr:.4f} dB")
print(f"SSIM         : {avg_ssim:.4f}")
print(f"SAM          : {avg_sam:.4f}°")

print("=" * 60)
print("TEST EVALUATION COMPLETE")
print("=" * 60)