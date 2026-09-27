from pathlib import Path
from io import BytesIO
import time

import pandas as pd
import pyarrow.parquet as pq


TACO_PATH = Path("data/sen2naipv2-crosssensor.taco")
SPLIT_FILE = Path("sen2naipv2_crosssensor_split.csv")
OUTPUT_ROOT = Path("data/sen2naipv2_extracted")


def read_metadata(f, parent_offset):
    """Read the small Parquet metadata block for one TACO sample."""

    f.seek(parent_offset)

    header = f.read(18)

    if header[:2] not in {b"#y", b"WX"}:
        raise ValueError(f"Invalid TACO block at offset {parent_offset}")

    footer_offset = int.from_bytes(header[2:10], "little") + parent_offset
    footer_length = int.from_bytes(header[10:18], "little")

    f.seek(footer_offset)
    footer = f.read(footer_length)

    return pq.read_table(BytesIO(footer)).to_pandas()


def extract_sample(f, parent_row, output_dir):
    """Extract LR and HR GeoTIFFs directly from the TACO file."""

    sample_id = parent_row["tortilla:id"]
    parent_offset = int(parent_row["tortilla:offset"])

    metadata = read_metadata(f, parent_offset)

    for _, row in metadata.iterrows():

        kind = row["tortilla:id"]

        if kind not in {"lr", "hr"}:
            continue

        relative_offset = int(row["tortilla:offset"])
        length = int(row["tortilla:length"])

        absolute_offset = parent_offset + relative_offset

        f.seek(absolute_offset)
        raster_bytes = f.read(length)

        output_path = output_dir / f"{sample_id}_{kind}.tif"

        # Resume support:
        # If the file already exists and has the correct size,
        # don't extract it again.
        if output_path.exists() and output_path.stat().st_size == length:
            continue

        output_path.write_bytes(raster_bytes)


def main():

    print("=" * 70)
    print("SEN2NAIPv2 DIRECT TACO EXTRACTOR")
    print("=" * 70)

    print("\nLoading TACO metadata...")

    # We only need the metadata table here.
    from data.dataset import load_taco_crosssensor

    taco = load_taco_crosssensor()

    print(f"Total TACO samples: {len(taco):,}")

    print("\nLoading split file...")

    split_df = pd.read_csv(SPLIT_FILE)

    print(split_df["split"].value_counts())

    # Map sample ID -> split
    split_map = dict(
        zip(
            split_df["tortilla:id"],
            split_df["split"]
        )
    )

    # Keep only samples that exist in our split file
    selected = taco[taco["tortilla:id"].isin(split_map)].copy()

    selected["split"] = selected["tortilla:id"].map(split_map)

    print(f"\nSamples selected: {len(selected):,}")

    print("\nSplit counts:")
    print(selected["split"].value_counts())

    # Create directories
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / split / "lr").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split / "hr").mkdir(parents=True, exist_ok=True)

    print("\nStarting direct extraction...")
    print("Source:", TACO_PATH)
    print("Output:", OUTPUT_ROOT)

    start_time = time.time()

    completed = 0
    skipped = 0
    failed = 0

    with open(TACO_PATH, "rb") as f:

        for position, (_, row) in enumerate(selected.iterrows(), start=1):

            sample_id = row["tortilla:id"]
            split = row["split"]

            lr_path = OUTPUT_ROOT / split / "lr" / f"{sample_id}_lr.tif"
            hr_path = OUTPUT_ROOT / split / "hr" / f"{sample_id}_hr.tif"

            # Check whether both files already exist
            if lr_path.exists() and hr_path.exists():
                skipped += 1
                completed += 1
                continue

            try:

                output_dir_lr = OUTPUT_ROOT / split / "lr"
                output_dir_hr = OUTPUT_ROOT / split / "hr"

                metadata = read_metadata(
                    f,
                    int(row["tortilla:offset"])
                )

                for _, asset in metadata.iterrows():

                    kind = asset["tortilla:id"]

                    if kind not in {"lr", "hr"}:
                        continue

                    relative_offset = int(asset["tortilla:offset"])
                    length = int(asset["tortilla:length"])

                    absolute_offset = (
                        int(row["tortilla:offset"])
                        + relative_offset
                    )

                    f.seek(absolute_offset)
                    raster_bytes = f.read(length)

                    if kind == "lr":
                        output_path = (
                            output_dir_lr /
                            f"{sample_id}_lr.tif"
                        )
                    else:
                        output_path = (
                            output_dir_hr /
                            f"{sample_id}_hr.tif"
                        )

                    output_path.write_bytes(raster_bytes)

                completed += 1

            except Exception as e:

                failed += 1

                print(
                    f"\nERROR [{position}/{len(selected)}] "
                    f"{sample_id}: {e}"
                )

                continue

            # Progress
            if position % 25 == 0 or position == 1:

                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0

                remaining = len(selected) - position
                eta_seconds = remaining / speed if speed > 0 else 0

                eta_minutes = eta_seconds / 60

                print(
                    f"{position:5d}/{len(selected)} | "
                    f"{speed:.2f} samples/sec | "
                    f"ETA: {eta_minutes:.1f} min | "
                    f"failed: {failed}"
                )

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)

    print(f"Processed : {completed:,}")
    print(f"Skipped   : {skipped:,}")
    print(f"Failed    : {failed:,}")
    print(f"Time      : {elapsed / 60:.2f} minutes")

    if elapsed > 0:
        print(f"Speed     : {completed / elapsed:.2f} samples/sec")

    print("\nOutput:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()