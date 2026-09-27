import os
import rasterio
from pathlib import Path

from data.dataset import load_taco_crosssensor, SEN2NAIPDataset


# Load local TACO
taco = load_taco_crosssensor()

ds = SEN2NAIPDataset(taco_dataset=taco, split="train", split_file="sen2naipv2_crosssensor_split.csv")

output_dir = Path("data/extract_test")
lr_dir = output_dir / "lr"
hr_dir = output_dir / "hr"

lr_dir.mkdir(parents=True, exist_ok=True)
hr_dir.mkdir(parents=True, exist_ok=True)

# Extract first 10 samples
for i in range(10):
    item = ds[i]

    lr = item["lr"].numpy()
    hr = item["hr"].numpy()
    sample_id = item["sample_id"]

    lr_path = lr_dir / f"{i:03d}_{sample_id}_lr.tif"
    hr_path = hr_dir / f"{i:03d}_{sample_id}_hr.tif"

    # Save LR
    with rasterio.open(
        lr_path,
        "w",
        driver="GTiff",
        height=lr.shape[1],
        width=lr.shape[2],
        count=lr.shape[0],
        dtype=lr.dtype,
    ) as dst:
        dst.write(lr)

    # Save HR
    with rasterio.open(
        hr_path,
        "w",
        driver="GTiff",
        height=hr.shape[1],
        width=hr.shape[2],
        count=hr.shape[0],
        dtype=hr.dtype,
    ) as dst:
        dst.write(hr)

    print(f"{i+1}/10 extracted")

print("\nDone!")
print("Output:", output_dir)