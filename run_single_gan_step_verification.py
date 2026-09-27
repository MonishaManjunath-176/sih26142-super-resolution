import sys
import torch
import torch.nn as nn
from data.dataset import SEN2NAIPDataset
from data.transforms import get_transforms
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.loss import GeneratorLoss, DiscriminatorLoss

def main():
    print("=== STEP 5: VERIFY ONE LR/HR PAIR FROM DATASET ===")
    transform = get_transforms(is_train=True, scale=4)
    dataset = SEN2NAIPDataset(transform=transform)
    print(f"Total dataset items: {len(dataset)}")

    sample = dataset[0]
    lr = sample["lr"]  # Shape: (4, 130, 130)
    hr = sample["hr"]  # Shape: (4, 520, 520)

    print("\n[LR Sample Details]")
    print(f" - Shape: {lr.shape}")
    print(f" - Dtype: {lr.dtype}")
    print(f" - Channels: {lr.shape[0]}")
    print(f" - Min Value: {lr.min().item():.4f}")
    print(f" - Max Value: {lr.max().item():.4f}")

    print("\n[HR Sample Details]")
    print(f" - Shape: {hr.shape}")
    print(f" - Dtype: {hr.dtype}")
    print(f" - Channels: {hr.shape[0]}")
    print(f" - Min Value: {hr.min().item():.4f}")
    print(f" - Max Value: {hr.max().item():.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device for verification: {device}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for this verification step.")
        sys.exit(1)

    torch.cuda.reset_peak_memory_stats()

    # Move tensors to GPU with batch size = 1
    lr_b = lr.unsqueeze(0).to(device)  # (1, 4, 130, 130)
    hr_b = hr.unsqueeze(0).to(device)  # (1, 4, 520, 520)

    print("\n=== STEP 7: SINGLE GENERATOR FORWARD PASS VERIFICATION ===")
    net_g = UNetGenerator(in_channels=4, out_channels=4, base_features=64, upscale_factor=4).to(device)
    net_g.eval()

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            sr_out = net_g(lr_b)

    print(f"Input LR Shape: {lr_b.shape}")
    print(f"Generator Output SR Shape: {sr_out.shape}")
    assert sr_out.shape == (1, 4, 520, 520), f"Expected (1, 4, 520, 520), got {sr_out.shape}"
    print("[SUCCESS] Generator produced exact 4x spatial upscaled tensor (1, 4, 520, 520)!")

    print("\n=== STEP 8: ONE COMPLETE GAN TRAINING STEP ON GPU (batch_size=1, AMP enabled) ===")
    net_d = PatchGANDiscriminator(in_channels=4, base_features=64).to(device)
    
    opt_g = torch.optim.Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=0.0001, betas=(0.5, 0.999))

    criterion_g = GeneratorLoss(lambda_l1=1.0, lambda_adv=0.01, lambda_spectral=0.1).to(device)
    criterion_d = DiscriminatorLoss().to(device)

    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    net_g.train()
    net_d.train()

    # 1. Discriminator Update Step
    opt_d.zero_grad()
    with torch.cuda.amp.autocast():
        sr_imgs = net_g(lr_b)
        pred_real = net_d(hr_b)
        pred_fake = net_d(sr_imgs.detach())
        loss_d, d_info = criterion_d(pred_real, pred_fake)

    scaler_d.scale(loss_d).backward()
    scaler_d.step(opt_d)
    scaler_d.update()

    # 2. Generator Update Step
    opt_g.zero_grad()
    with torch.cuda.amp.autocast():
        pred_fake_for_g = net_d(sr_imgs)
        loss_g, g_info = criterion_g(pred_fake_for_g, sr_imgs, hr_b)

    scaler_g.scale(loss_g).backward()
    scaler_g.step(opt_g)
    scaler_g.update()

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_vram_gb = peak_vram_mb / 1024.0

    print("\n=== SINGLE GAN STEP VERIFICATION RESULTS ===")
    print(f"Discriminator Loss: {loss_d.item():.4f} (Real: {d_info['d_real']:.4f}, Fake: {d_info['d_fake']:.4f})")
    print(f"Generator Loss:     {loss_g.item():.4f} (L1: {g_info['l1_loss']:.4f}, Adv: {g_info['adv_loss']:.4f}, SAM: {g_info['spectral_loss']:.4f})")
    print(f"Peak GPU VRAM Allocated: {peak_vram_mb:.2f} MB ({peak_vram_gb:.2f} GB)")
    print("[SUCCESS] ONE COMPLETE GAN TRAINING STEP SUCCEEDED ON GPU WITHOUT OOM!")

if __name__ == "__main__":
    main()
