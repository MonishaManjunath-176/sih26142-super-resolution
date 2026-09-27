import os
from huggingface_hub import hf_hub_download
import rasterio
import torch

save_dir = "data/sample_subset"
os.makedirs(save_dir, exist_ok=True)

print("=== DOWNLOADING TINY OFFICIAL DEMO SUBSET FROM isp-uv-es/SEN2NAIP ===")

lr_path = hf_hub_download(
    repo_id="isp-uv-es/SEN2NAIP",
    filename="demo/cross-sensor/ROI_0000/lr.tif",
    repo_type="dataset",
    local_dir=save_dir
)

hr_path = hf_hub_download(
    repo_id="isp-uv-es/SEN2NAIP",
    filename="demo/cross-sensor/ROI_0000/hr.tif",
    repo_type="dataset",
    local_dir=save_dir
)

print(f"Downloaded LR TIFF to: {lr_path}")
print(f"Downloaded HR TIFF to: {hr_path}")

print("\n=== STEP 5: VERIFYING LR / HR TENSORS ===")

with rasterio.open(lr_path) as src_lr:
    lr_data = src_lr.read() # Shape: (C, H, W)
    lr_tensor = torch.from_numpy(lr_data).float()

with rasterio.open(hr_path) as src_hr:
    hr_data = src_hr.read() # Shape: (C, H, W)
    hr_tensor = torch.from_numpy(hr_data).float()

print("LR Image Details:")
print(f" - Shape: {lr_tensor.shape}")
print(f" - Dtype: {lr_tensor.dtype}")
print(f" - Channels: {lr_tensor.shape[0]}")
print(f" - Min Value: {lr_tensor.min().item():.4f}")
print(f" - Max Value: {lr_tensor.max().item():.4f}")

print("\nHR Image Details:")
print(f" - Shape: {hr_tensor.shape}")
print(f" - Dtype: {hr_tensor.dtype}")
print(f" - Channels: {hr_tensor.shape[0]}")
print(f" - Min Value: {hr_tensor.min().item():.4f}")
print(f" - Max Value: {hr_tensor.max().item():.4f}")
