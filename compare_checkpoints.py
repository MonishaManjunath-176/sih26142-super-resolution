from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from models.generator import CNNAttentionGenerator


# ============================================================
# SETTINGS
# ============================================================

LR_PATH = Path(
    "data/sen2naipv2_extracted/test/lr/"
    "na5120_e1185n0741__m_3912364_sw_10_060_20200604_lr.tif"
)

CHECKPOINT_DIR = Path("outputs/checkpoints")

OUTPUT_PATH = Path(
    "outputs/checkpoint_comparison.png"
)

EPOCHS = [1, 10, 20, 30]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# READ IMAGE
# ============================================================

def read_tiff(path):

    with rasterio.open(path) as src:
        image = src.read(
            [1, 2, 3, 4]
        ).astype(np.float32)

    max_value = np.nanmax(image)

    if max_value > 255:
        image /= 10000.0
    elif max_value > 1:
        image /= 255.0

    return np.clip(image, 0.0, 1.0)


# ============================================================
# RGB DISPLAY
# ============================================================

def make_rgb(image):

    rgb = np.transpose(
        image[:3],
        (1, 2, 0)
    )

    output = np.zeros_like(rgb)

    for c in range(3):

        band = rgb[:, :, c]

        low = np.percentile(
            band, 2
        )

        high = np.percentile(
            band, 98
        )

        if high > low:
            band = (
                band - low
            ) / (
                high - low
            )

        output[:, :, c] = np.clip(
            band,
            0,
            1
        )

    return output


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path):

    model = CNNAttentionGenerator(
        in_channels=4,
        out_channels=4,
        base_features=64,
        upscale_factor=4,
        num_res_blocks=4,
        attention_reduction=8,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if "generator_state_dict" in checkpoint:
        state_dict = checkpoint[
            "generator_state_dict"
        ]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint[
            "model_state_dict"
        ]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint[
            "state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SIH26142 — CHECKPOINT ARTIFACT ANALYSIS")
print("=" * 70)

print("Device:", DEVICE)
print("Input:", LR_PATH)
print("Epochs:", EPOCHS)
print()


# Read LR image
lr_np = read_tiff(LR_PATH)

# IMPORTANT:
# Training uses [-1, 1]
lr_norm = (
    lr_np * 2.0
) - 1.0

lr_tensor = torch.from_numpy(
    lr_norm
).unsqueeze(0).to(DEVICE)


results = []


with torch.no_grad():

    for epoch in EPOCHS:

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"generator_epoch_{epoch}.pth"
        )

        print(
            f"Evaluating epoch {epoch}..."
        )

        model = load_model(
            checkpoint_path
        )

        sr = model(
            lr_tensor
        )

        # [-1,1] -> [0,1]
        sr = (
            sr.squeeze(0)
            .cpu()
            .numpy()
        )

        sr = (
            sr + 1.0
        ) / 2.0

        sr = np.clip(
            sr,
            0,
            1
        )

        results.append(
            sr
        )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(22, 5)
)


# Original LR
axes[0].imshow(
    make_rgb(lr_np)
)

axes[0].set_title(
    "Input\nSentinel-2 LR"
)


# Epoch outputs
for i, epoch in enumerate(EPOCHS):

    axes[i + 1].imshow(
        make_rgb(results[i])
    )

    axes[i + 1].set_title(
        f"Epoch {epoch}\nCNN + CBAM"
    )


for ax in axes:
    ax.axis("off")


fig.suptitle(
    "SIH26142 — Artifact Evolution Across Training",
    fontsize=16
)

plt.tight_layout()

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_PATH
)