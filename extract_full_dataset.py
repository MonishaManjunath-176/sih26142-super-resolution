import os
import time
import rasterio
from pathlib import Path

from data.dataset import load_taco_crosssensor, SEN2NAIPDataset


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SPLITS = {
    "train": 6377,
    "val": 807,
    "test": 816,
}

SPLIT_FILE = "sen2naipv2_crosssensor_split.csv"
OUTPUT_ROOT = Path("data/expanded")


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("Loading SEN2NAIPv2...")

taco = load_taco_crosssensor()

print("TACO loaded.")


# --------------------------------------------------
# EXTRACT EACH SPLIT
# --------------------------------------------------

for split, expected_count in SPLITS.items():

    print("\n" + "=" * 60)
    print(f"EXTRACTING: {split.upper()}")
    print(f"Expected samples: {expected_count}")
    print("=" * 60)

    dataset = SEN2NAIPDataset(
        taco_dataset=taco,
        split=split,
        split_file=SPLIT_FILE,
    )

    lr_dir = OUTPUT_ROOT / split / "lr"
    hr_dir = OUTPUT_ROOT / split / "hr"

    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)

    print(f"Dataset contains: {total} samples")

    start_time = time.time()

    for i in range(total):

        item = dataset[i]

        lr = item["lr"].numpy()
        hr = item["hr"].numpy()
        sample_id = item["sample_id"]

        lr_path = lr_dir / f"{i:05d}_{sample_id}_lr.tif"
        hr_path = hr_dir / f"{i:05d}_{sample_id}_hr.tif"

        # Write LR
        with rasterio.open(
            lr_path,
            "w",
            driver="GTiff",
            height=lr.shape[1],
            width=lr.shape[2],
            count=lr.shape[0],
            dtype="float32",
        ) as dst:
            dst.write(lr)

        # Write HR
        with rasterio.open(
            hr_path,
            "w",
            driver="GTiff",
            height=hr.shape[1],
            width=hr.shape[2],
            count=hr.shape[0],
            dtype="float32",
        ) as dst:
            dst.write(hr)

        # Progress every 100 samples
        if (i + 1) % 100 == 0 or i == 0:

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0

            print(
                f"{i + 1}/{total} | "
                f"{rate:.2f} samples/sec | "
                f"ETA: {remaining / 60:.1f} min"
            )

    elapsed = time.time() - start_time

    print(f"\nFinished {split}!")
    print(f"Time: {elapsed / 60:.1f} minutes")


print("\n" + "=" * 60)
print("ALL EXTRACTION COMPLETE")
print("=" * 60)
print(f"Output: {OUTPUT_ROOT}")