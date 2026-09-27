from pathlib import Path
from io import BytesIO
import pyarrow.parquet as pq
import rasterio
from data.dataset import load_taco_crosssensor

TACO_PATH = Path("data/sen2naipv2-crosssensor.taco")

taco = load_taco_crosssensor()
parent = taco.iloc[0]

parent_offset = int(parent["tortilla:offset"])

# --------------------------------------------------
# Read TACO header
# --------------------------------------------------
with open(TACO_PATH, "rb") as f:
    f.seek(parent_offset)
    header = f.read(18)

    footer_offset = int.from_bytes(header[2:10], "little") + parent_offset
    footer_length = int.from_bytes(header[10:18], "little")

    # --------------------------------------------------
    # Read Parquet metadata
    # --------------------------------------------------
    f.seek(footer_offset)
    footer = f.read(footer_length)

    metadata = pq.read_table(BytesIO(footer)).to_pandas()

    print("\nSample:", parent["tortilla:id"])
    print(metadata[["tortilla:id", "tortilla:offset", "tortilla:length"]])

    # --------------------------------------------------
    # Extract LR + HR directly
    # --------------------------------------------------
    output_dir = Path("data/direct_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in metadata.iterrows():

        name = row["tortilla:id"]
        relative_offset = int(row["tortilla:offset"])
        length = int(row["tortilla:length"])

        absolute_offset = parent_offset + relative_offset

        print(
            f"\n{name.upper()}: "
            f"absolute offset={absolute_offset}, "
            f"length={length:,}"
        )

        f.seek(absolute_offset)
        raster_bytes = f.read(length)

        output_path = output_dir / f"{name}.tif"
        output_path.write_bytes(raster_bytes)

        print("Saved:", output_path)


# --------------------------------------------------
# Verify the extracted GeoTIFFs
# --------------------------------------------------
print("\n" + "=" * 60)
print("VERIFYING")
print("=" * 60)

for name in ["lr", "hr"]:

    path = Path("data/direct_test") / f"{name}.tif"

    with rasterio.open(path) as src:

        print(
            f"{name.upper()}: "
            f"shape={src.shape}, "
            f"bands={src.count}, "
            f"crs={src.crs}, "
            f"resolution={src.res}"
        )