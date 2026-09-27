from pathlib import Path
from data.dataset import load_taco_crosssensor

TACO_PATH = Path("data/sen2naipv2-crosssensor.taco")

taco = load_taco_crosssensor()

row = taco.iloc[0]

offset = int(row["tortilla:offset"])
length = int(row["tortilla:length"])

print("Offset:", offset)
print("Length:", length)

with open(TACO_PATH, "rb") as f:
    f.seek(offset)
    data = f.read(length)

print("Bytes read:", len(data))
print("First 32 bytes:", data[:32])
print("Last 32 bytes:", data[-32:])

# Save one raw sample block for inspection
Path("data/sample_block.bin").write_bytes(data)

print("Saved: data/sample_block.bin")