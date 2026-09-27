from pathlib import Path
from io import BytesIO
import pyarrow.parquet as pq
from data.dataset import load_taco_crosssensor

TACO_PATH = Path("data/sen2naipv2-crosssensor.taco")

taco = load_taco_crosssensor()
row = taco.iloc[0]

offset = int(row["tortilla:offset"])

with open(TACO_PATH, "rb") as f:
    f.seek(offset)

    header = f.read(18)

    footer_offset = int.from_bytes(header[2:10], "little") + offset
    footer_length = int.from_bytes(header[10:18], "little")

    f.seek(footer_offset)
    footer = f.read(footer_length)

table = pq.read_table(BytesIO(footer))
df = table.to_pandas()

print("\n" + "=" * 80)
print("FOOTER ROWS")
print("=" * 80)

print(df.to_string(index=False))

print("\n" + "=" * 80)
print("ROW DETAILS")
print("=" * 80)

for i, r in df.iterrows():
    print(f"\n--- Row {i} ---")
    for column in df.columns:
        print(f"{column}: {r[column]}")

print("\nDone.")