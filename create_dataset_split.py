import tacoreader
import pandas as pd
import numpy as np

# --------------------------------------------------
# Load SEN2NAIPv2 Cross-Sensor dataset
# --------------------------------------------------

print("[1/5] Loading dataset...")
dataset = tacoreader.load(
    "tacofoundation:sen2naipv2-crosssensor"
)

df = dataset.copy()

print(f"Total samples: {len(df)}")
print(f"Unique locations: {df['stac:centroid'].nunique()}")

# --------------------------------------------------
# Create reproducible location groups
# --------------------------------------------------

locations = df["stac:centroid"].drop_duplicates().tolist()

rng = np.random.default_rng(42)
rng.shuffle(locations)

n_locations = len(locations)

n_train = int(0.80 * n_locations)
n_val = int(0.10 * n_locations)

train_locations = set(locations[:n_train])
val_locations = set(locations[n_train:n_train + n_val])
test_locations = set(locations[n_train + n_val:])

# --------------------------------------------------
# Assign each sample to a split
# --------------------------------------------------

def assign_split(centroid):
    if centroid in train_locations:
        return "train"
    elif centroid in val_locations:
        return "val"
    else:
        return "test"


df["split"] = df["stac:centroid"].apply(assign_split)

# --------------------------------------------------
# Save sample IDs + useful metadata
# --------------------------------------------------

output = df[
    [
        "tortilla:id",
        "stac:centroid",
        "split"
    ]
].copy()

output.to_csv(
    "sen2naipv2_crosssensor_split.csv",
    index=False
)

# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n[5/5] Split created successfully!")

print("\nSample counts:")
print(output["split"].value_counts())

print("\nLocation counts:")
print(
    output.groupby("split")["stac:centroid"]
    .nunique()
)

print("\nSaved:")
print("sen2naipv2_crosssensor_split.csv")

# Verify that no centroid appears in multiple splits
location_split_counts = (
    output.groupby("stac:centroid")["split"]
    .nunique()
)

leakage = (location_split_counts > 1).sum()

print(f"\nLocation leakage check: {leakage}")

if leakage == 0:
    print("✓ No geographic-location leakage detected.")
else:
    print("WARNING: Some locations appear in multiple splits.")