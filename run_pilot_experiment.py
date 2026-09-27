import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SEN2NAIPDataset
from data.transforms import get_transforms
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.loss import GeneratorLoss, DiscriminatorLoss
from evaluation.metrics import calculate_psnr, calculate_ssim, calculate_sam, plot_visual_comparison


def run_pilot():
    print("=================================================================")
    print("   STARTING PILOT TRAINING EXPERIMENT (4 Epochs, Batch Size 1)   ")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using acceleration device: {device}")
    if not torch.cuda.is_available():
        print("[Error] CUDA device is required for GPU pilot experiment.")
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats()

    # Dataloaders
    train_transform = get_transforms(is_train=True, scale=4)
    val_transform = get_transforms(is_train=False, scale=4)

    train_dataset = SEN2NAIPDataset(lr_dir="data/train/lr", hr_dir="data/train/hr", transform=train_transform, is_train=True)
    val_dataset = SEN2NAIPDataset(lr_dir="data/val/lr", hr_dir="data/val/hr", transform=val_transform, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"[Dataset] Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Initialize Models
    net_g = UNetGenerator(in_channels=4, out_channels=4, base_features=64, upscale_factor=4).to(device)
    net_d = PatchGANDiscriminator(in_channels=4, base_features=64).to(device)

    # Optimizers
    opt_g = torch.optim.Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=0.0001, betas=(0.5, 0.999))

    # Loss Functions & AMP Scalers
    criterion_g = GeneratorLoss(lambda_l1=1.0, lambda_adv=0.01, lambda_spectral=0.1).to(device)
    criterion_d = DiscriminatorLoss().to(device)

    scaler_g = torch.amp.GradScaler('cuda')
    scaler_d = torch.amp.GradScaler('cuda')

    # Create directories
    ckpt_dir = "outputs/checkpoints"
    sample_dir = "outputs/samples"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    total_epochs = 4
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, total_epochs + 1):
        net_g.train()
        net_d.train()

        running_g_loss = 0.0
        running_d_loss = 0.0
        running_l1_loss = 0.0
        running_adv_loss = 0.0
        running_sam_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
        for step, batch in enumerate(pbar):
            lr_imgs = batch["lr"].to(device)  # (1, 4, 130, 130)
            hr_imgs = batch["hr"].to(device)  # (1, 4, 520, 520)

            # ---------------------
            # Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            with torch.amp.autocast('cuda'):
                sr_imgs = net_g(lr_imgs)
                pred_real = net_d(hr_imgs)
                pred_fake = net_d(sr_imgs.detach())
                loss_d, d_info = criterion_d(pred_real, pred_fake)

            # Check for NaN / Inf
            if torch.isnan(loss_d) or torch.isinf(loss_d):
                print(f"\n[ERROR] NaN/Inf detected in Discriminator loss at Epoch {epoch}, Step {step}")
                sys.exit(1)

            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ---------------------
            # Train Generator
            # ---------------------
            opt_g.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_fake_for_g = net_d(sr_imgs)
                loss_g, g_info = criterion_g(pred_fake_for_g, sr_imgs, hr_imgs)

            if torch.isnan(loss_g) or torch.isinf(loss_g):
                print(f"\n[ERROR] NaN/Inf detected in Generator loss at Epoch {epoch}, Step {step}")
                sys.exit(1)

            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            running_g_loss += loss_g.item()
            running_d_loss += loss_d.item()
            running_l1_loss += g_info["l1_loss"]
            running_adv_loss += g_info["adv_loss"]
            running_sam_loss += g_info["spectral_loss"]

            pbar.set_postfix({
                "G_Loss": f"{loss_g.item():.2f}",
                "D_Loss": f"{loss_d.item():.2f}",
                "L1": f"{g_info['l1_loss']:.3f}"
            })

        avg_g_loss = running_g_loss / len(train_loader)
        avg_d_loss = running_d_loss / len(train_loader)
        avg_l1 = running_l1_loss / len(train_loader)
        avg_adv = running_adv_loss / len(train_loader)
        avg_sam = running_sam_loss / len(train_loader)

        # ---------------------
        # Validation Step
        # ---------------------
        net_g.eval()
        val_psnr_list = []
        val_ssim_list = []
        val_sam_list = []
        val_reconstruction_list = []

        with torch.no_grad():
            for idx, val_batch in enumerate(val_loader):
                val_lr = val_batch["lr"].to(device)
                val_hr = val_batch["hr"].to(device)

                with torch.amp.autocast('cuda'):
                    val_sr = net_g(val_lr)

                val_lr_cpu = val_lr.squeeze(0).cpu()
                val_sr_cpu = val_sr.squeeze(0).cpu()
                val_hr_cpu = val_hr.squeeze(0).cpu()

                psnr_val = calculate_psnr(val_sr_cpu, val_hr_cpu)
                ssim_val = calculate_ssim(val_sr_cpu, val_hr_cpu)
                sam_val = calculate_sam(val_sr_cpu, val_hr_cpu)

                val_psnr_list.append(psnr_val)
                val_ssim_list.append(ssim_val)
                val_sam_list.append(sam_val)
                val_reconstruction_list.append(
                    (torch.nn.functional.l1_loss(val_sr, val_hr)
                     + 0.1 * criterion_g.spectral_loss(val_sr, val_hr)).item()
                )

                # Save comparison plot for first val image
                if idx == 0:
                    sample_path = os.path.join(sample_dir, f"epoch_{epoch}_comparison.png")
                    plot_visual_comparison(val_lr_cpu, val_sr_cpu, val_hr_cpu, save_path=sample_path)

        mean_psnr = float(np.mean(val_psnr_list))
        mean_ssim = float(np.mean(val_ssim_list))
        mean_sam_eval = float(np.mean(val_sam_list))
        mean_val_loss = float(np.mean(val_reconstruction_list))

        # Save Checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"generator_epoch_{epoch}.pth")
        torch.save(net_g.state_dict(), ckpt_path)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(net_g.state_dict(), os.path.join(ckpt_dir, "best_generator.pth"))
            print("[Info] Saved best_generator.pth from the lowest validation reconstruction/spectral loss.")

        epoch_summary = {
            "epoch": epoch,
            "g_loss": avg_g_loss,
            "d_loss": avg_d_loss,
            "l1_loss": avg_l1,
            "adv_loss": avg_adv,
            "sam_train": avg_sam,
            "val_psnr": mean_psnr,
            "val_ssim": mean_ssim,
            "val_sam": mean_sam_eval,
            "val_loss": mean_val_loss,
            "ckpt": ckpt_path
        }
        history.append(epoch_summary)

        print(f"\n--- Epoch {epoch} Metrics Summary ---")
        print(f"  • Train G Loss:     {avg_g_loss:.4f} (L1: {avg_l1:.4f}, Adv: {avg_adv:.4f}, SAM: {avg_sam:.4f})")
        print(f"  • Train D Loss:     {avg_d_loss:.4f}")
        print(f"  • Val PSNR:         {mean_psnr:.2f} dB")
        print(f"  • Val SSIM:         {mean_ssim:.4f}")
        print(f"  • Val SAM Angle:    {mean_sam_eval:.2f}°")
        print(f"  • Val Selection Loss: {mean_val_loss:.4f}")
        print(f"  • Checkpoint saved: {ckpt_path}")
        print(f"  • Sample comparison: outputs/samples/epoch_{epoch}_comparison.png\n")

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_gb = peak_vram_mb / 1024.0

    print("=================================================================")
    print("            PILOT TRAINING EXPERIMENT COMPLETED                  ")
    print("=================================================================")
    print(f"Peak GPU VRAM Allocated: {peak_vram_mb:.2f} MB ({peak_vram_gb:.2f} GB)")
    print("No OOM, NaN, or exploding loss errors encountered!")

if __name__ == "__main__":
    run_pilot()
