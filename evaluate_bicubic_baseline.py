import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import SEN2NAIPDataset
from data.transforms import get_transforms
from evaluation.metrics import calculate_psnr, calculate_ssim, calculate_sam


def main():
    print("=================================================================")
    print("   EVALUATING BICUBIC INTERPOLATION BASELINE ON VAL DATASET      ")
    print("=================================================================")

    out_dir = "outputs/baselines"
    os.makedirs(out_dir, exist_ok=True)

    # Load exact 10 validation samples
    val_transform = get_transforms(is_train=False, scale=4)
    val_dataset = SEN2NAIPDataset(lr_dir="data/val/lr", hr_dir="data/val/hr", transform=val_transform, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"[Dataset] Loaded {len(val_dataset)} validation samples.")

    per_sample_results = []

    for idx, batch in enumerate(val_loader):
        lr_tensor = batch["lr"]  # (1, 4, 130, 130)
        hr_tensor = batch["hr"]  # (1, 4, 520, 520)

        # 4x Bicubic Interpolation from 130x130 to 520x520
        bicubic_tensor = F.interpolate(lr_tensor, size=(520, 520), mode='bicubic', align_corners=False)

        # Squeeze batch dimension to (4, 520, 520)
        bicubic_img = bicubic_tensor.squeeze(0)
        hr_img = hr_tensor.squeeze(0)

        # Calculate metrics
        psnr_val = calculate_psnr(bicubic_img, hr_img)
        ssim_val = calculate_ssim(bicubic_img, hr_img)
        sam_val = calculate_sam(bicubic_img, hr_img)

        sample_info = {
            "Sample_ID": idx + 1,
            "Bicubic_PSNR_dB": round(psnr_val, 4),
            "Bicubic_SSIM": round(ssim_val, 4),
            "Bicubic_SAM_deg": round(sam_val, 4)
        }
        per_sample_results.append(sample_info)

    df_samples = pd.DataFrame(per_sample_results)

    # Calculate Mean values
    mean_psnr = float(df_samples["Bicubic_PSNR_dB"].mean())
    mean_ssim = float(df_samples["Bicubic_SSIM"].mean())
    mean_sam = float(df_samples["Bicubic_SAM_deg"].mean())

    # Add Summary Mean row to DataFrame
    summary_row = {
        "Sample_ID": "MEAN",
        "Bicubic_PSNR_dB": round(mean_psnr, 4),
        "Bicubic_SSIM": round(mean_ssim, 4),
        "Bicubic_SAM_deg": round(mean_sam, 4)
    }

    df_csv = pd.concat([df_samples, pd.DataFrame([summary_row])], ignore_index=True)

    # Save to outputs/baselines/bicubic_metrics.csv
    csv_path = os.path.join(out_dir, "bicubic_metrics.csv")
    df_csv.to_csv(csv_path, index=False)
    print(f"[Saved] Per-sample & mean metrics saved to: {csv_path}")

    # Save to outputs/baselines/bicubic_summary.txt
    txt_path = os.path.join(out_dir, "bicubic_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=== BICUBIC INTERPOLATION BASELINE SUMMARY (10 Val Samples) ===\n\n")
        f.write(f" • Total Validation Samples Evaluated: {len(val_dataset)}\n")
        f.write(f" • Mean PSNR (dB) : {mean_psnr:.2f} dB\n")
        f.write(f" • Mean SSIM      : {mean_ssim:.4f}\n")
        f.write(f" • Mean SAM Angle : {mean_sam:.2f}°\n\n")
        f.write("Per-Sample Details:\n")
        for _, r in df_samples.iterrows():
            f.write(f"   Sample {int(r['Sample_ID']):02d}: PSNR = {r['Bicubic_PSNR_dB']:.2f} dB | SSIM = {r['Bicubic_SSIM']:.4f} | SAM = {r['Bicubic_SAM_deg']:.2f}°\n")

    print(f"[Saved] Summary text saved to: {txt_path}")

    print("\n=================================================================")
    print("          BICUBIC BASELINE EVALUATION COMPLETED                  ")
    print("=================================================================")
    print(f"Mean PSNR:  {mean_psnr:.2f} dB")
    print(f"Mean SSIM:  {mean_ssim:.4f}")
    print(f"Mean SAM:   {mean_sam:.2f}°")


if __name__ == "__main__":
    main()
