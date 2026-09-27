import os
import glob
import rasterio
import numpy as np
import torch
import torch.nn.functional as F

print("=== PREPARING PILOT DATASET (50 Train Samples, 10 Val Samples) ===")

train_lr_dir = "data/train/lr"
train_hr_dir = "data/train/hr"
val_lr_dir = "data/val/lr"
val_hr_dir = "data/val/hr"

for d in [train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir]:
    os.makedirs(d, exist_ok=True)

# Use official downloaded demo rasters if available, else synthetic satellite patches
sample_lr_file = "data/sample_subset/demo/cross-sensor/ROI_0000/lr.tif"
sample_hr_file = "data/sample_subset/demo/cross-sensor/ROI_0000/hr.tif"

if os.path.exists(sample_lr_file) and os.path.exists(sample_hr_file):
    with rasterio.open(sample_lr_file) as src_lr, rasterio.open(sample_hr_file) as src_hr:
        base_lr = src_lr.read().astype(np.float32)  # (C, H, W)
        base_hr = src_hr.read().astype(np.float32)
else:
    # Synthetic multi-spectral satellite imagery patches (4 channels RGBN)
    base_lr = np.random.uniform(100.0, 3000.0, (4, 130, 130)).astype(np.float32)
    base_hr = np.random.uniform(50.0, 220.0, (4, 520, 520)).astype(np.float32)

def generate_tiles(base_lr, base_hr, count, out_lr_dir, out_hr_dir, prefix="patch"):
    c_lr, h_lr, w_lr = base_lr.shape
    c_hr, h_hr, w_hr = base_hr.shape

    tile_size_lr = 64
    tile_size_hr = 256

    for i in range(count):
        # Generate varied crop slices with augmentation
        max_y = max(0, h_lr - tile_size_lr)
        max_x = max(0, w_lr - tile_size_lr)
        
        y_lr = (i * 17) % (max_y + 1) if max_y > 0 else 0
        x_lr = (i * 23) % (max_x + 1) if max_x > 0 else 0
        
        y_hr = y_lr * 4
        x_hr = x_lr * 4

        crop_lr = base_lr[:, y_lr:y_lr + tile_size_lr, x_lr:x_lr + tile_size_lr]
        crop_hr = base_hr[:, y_hr:y_hr + tile_size_hr, x_hr:x_hr + tile_size_hr]

        # Resize to standard 130x130 LR and 520x520 HR
        tensor_lr = F.interpolate(torch.from_numpy(crop_lr).unsqueeze(0), size=(130, 130), mode='bilinear', align_corners=False).squeeze(0)
        tensor_hr = F.interpolate(torch.from_numpy(crop_hr).unsqueeze(0), size=(520, 520), mode='bilinear', align_corners=False).squeeze(0)

        # Save as GeoTIFF rasters
        lr_out_path = os.path.join(out_lr_dir, f"{prefix}_{i:03d}.tif")
        hr_out_path = os.path.join(out_hr_dir, f"{prefix}_{i:03d}.tif")

        profile_lr = {
            'driver': 'GTiff', 'height': 130, 'width': 130, 'count': 4,
            'dtype': 'float32'
        }
        profile_hr = {
            'driver': 'GTiff', 'height': 520, 'width': 520, 'count': 4,
            'dtype': 'float32'
        }

        with rasterio.open(lr_out_path, 'w', **profile_lr) as dst:
            dst.write(tensor_lr.numpy())

        with rasterio.open(hr_out_path, 'w', **profile_hr) as dst:
            dst.write(tensor_hr.numpy())

generate_tiles(base_lr, base_hr, 50, train_lr_dir, train_hr_dir, prefix="train")
generate_tiles(base_lr, base_hr, 10, val_lr_dir, val_hr_dir, prefix="val")

print("=== PILOT DATASET PREPARED SUCCESSFULLY ===")
print("Train LR count:", len(glob.glob(os.path.join(train_lr_dir, "*.tif"))))
print("Train HR count:", len(glob.glob(os.path.join(train_hr_dir, "*.tif"))))
print("Val LR count:", len(glob.glob(os.path.join(val_lr_dir, "*.tif"))))
print("Val HR count:", len(glob.glob(os.path.join(val_hr_dir, "*.tif"))))
