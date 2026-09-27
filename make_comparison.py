import rasterio
import numpy as np
import matplotlib.pyplot as plt

files = [
    ("LR", "data/val/lr/val_000.tif"),
    ("SR (OUR MODEL)", "outputs/val_000_SR.tif"),
    ("HR", "data/val/hr/val_000.tif")
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (title, path) in zip(axes, files):
    with rasterio.open(path) as src:
        data = src.read([1, 2, 3]).astype(np.float32)

    rgb = []

    for band in data:
        low = np.percentile(band, 2)
        high = np.percentile(band, 98)
        band = (band - low) / (high - low + 1e-8)
        band = np.clip(band, 0, 1)
        rgb.append(band)

    rgb = np.stack(rgb, axis=-1)

    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()

output = "outputs/LR_SR_HR_comparison.png"
plt.savefig(output, dpi=200, bbox_inches="tight")
plt.close()

print("Saved:", output)