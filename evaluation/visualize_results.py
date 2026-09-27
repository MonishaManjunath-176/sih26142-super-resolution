from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from models.generator import CNNAttentionGenerator


# ============================================================
# SETTINGS
# ============================================================

TEST_LR_DIR = Path(
    "data/sen2naipv2_extracted/test/lr"
)

TEST_HR_DIR = Path(
    "data/sen2naipv2_extracted/test/hr"
)

CHECKPOINT = Path(
    "outputs/checkpoints/best_generator.pth"
)

OUTPUT_DIR = Path(
    "outputs/visual_comparisons"
)

NUM_IMAGES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("SIH26142 — VISUAL SUPER-RESOLUTION COMPARISON")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Checkpoint: {CHECKPOINT}")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

model = CNNAttentionGenerator(
    in_channels=4,
    out_channels=4,
    base_features=64,
    upscale_factor=4,
    num_res_blocks=4,
    attention_reduction=8,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "generator_state_dict" in checkpoint:
    state_dict = checkpoint["generator_state_dict"]

elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]

elif "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]

else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# READ TIFF
# ============================================================

def read_tiff(path):

    with rasterio.open(path) as src:

        data = src.read(
            [1, 2, 3, 4]
        ).astype(np.float32)

    max_value = np.nanmax(data)

    if max_value > 255:

        data /= 10000.0

    elif max_value > 1:

        data /= 255.0

    data = np.clip(
        data,
        0.0,
        1.0
    )

    return data


# ============================================================
# RGB DISPLAY
# ============================================================

def make_rgb(image):

    rgb = image[:3]

    rgb = np.transpose(
        rgb,
        (1, 2, 0)
    )

    output = np.zeros_like(rgb)

    for channel in range(3):

        band = rgb[:, :, channel]

        low = np.percentile(
            band,
            2
        )

        high = np.percentile(
            band,
            98
        )

        if high > low:

            band = (
                band - low
            ) / (
                high - low
            )

        else:

            band = np.zeros_like(
                band
            )

        output[:, :, channel] = np.clip(
            band,
            0,
            1
        )

    return output


# ============================================================
# FIND LR/HR PAIRS
# ============================================================

lr_files = sorted(
    TEST_LR_DIR.glob("*.tif")
)

hr_files = sorted(
    TEST_HR_DIR.glob("*.tif")
)

hr_lookup = {
    path.stem.replace(
        "_hr",
        ""
    ): path
    for path in hr_files
}

pairs = []

for lr_path in lr_files:

    sample_id = lr_path.stem.replace(
        "_lr",
        ""
    )

    if sample_id in hr_lookup:

        pairs.append(
            (
                lr_path,
                hr_lookup[sample_id]
            )
        )


if not pairs:

    raise RuntimeError(
        "No matching LR/HR pairs found."
    )


print(
    f"Found {len(pairs)} test pairs."
)

print(
    f"Creating {min(NUM_IMAGES, len(pairs))} comparisons..."
)

print()


# ============================================================
# SELECT SPREAD-OUT SAMPLES
# ============================================================

num_to_make = min(
    NUM_IMAGES,
    len(pairs)
)

indices = np.linspace(
    0,
    len(pairs) - 1,
    num_to_make,
    dtype=int
)

selected_pairs = [
    pairs[i]
    for i in indices
]


# ============================================================
# GENERATE VISUAL COMPARISONS
# ============================================================

with torch.no_grad():

    for image_number, (
        lr_path,
        hr_path
    ) in enumerate(
        selected_pairs,
        start=1
    ):

        print(
            f"[{image_number}/{num_to_make}] "
            f"{lr_path.stem}"
        )

        # ----------------------------------------------------
        # READ ORIGINAL IMAGES [0,1]
        # ----------------------------------------------------

        lr_np = read_tiff(
            lr_path
        )

        hr_np = read_tiff(
            hr_path
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # MODEL WAS TRAINED ON [-1,1]
        # ----------------------------------------------------

        lr_norm = (
            lr_np * 2.0
        ) - 1.0

        lr_tensor = torch.from_numpy(
            lr_norm
        ).unsqueeze(0).to(DEVICE)

        # ----------------------------------------------------
        # CNN + CBAM
        # ----------------------------------------------------

        sr_tensor = model(
            lr_tensor
        )

        # ----------------------------------------------------
        # CONVERT MODEL OUTPUT
        # [-1,1] -> [0,1]
        # ----------------------------------------------------

        sr_np = (
            sr_tensor
            .squeeze(0)
            .cpu()
            .numpy()
        )

        sr_np = (
            sr_np + 1.0
        ) / 2.0

        sr_np = np.clip(
            sr_np,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # BICUBIC BASELINE
        #
        # Bicubic is applied to the original [0,1] LR image.
        # ----------------------------------------------------

        lr_tensor_raw = torch.from_numpy(
            lr_np
        ).unsqueeze(0).to(DEVICE)

        bicubic_tensor = F.interpolate(
            lr_tensor_raw,
            size=hr_np.shape[-2:],
            mode="bicubic",
            align_corners=False
        )

        bicubic_np = (
            bicubic_tensor
            .squeeze(0)
            .cpu()
            .numpy()
        )

        bicubic_np = np.clip(
            bicubic_np,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # RGB PREVIEWS
        # ----------------------------------------------------

        lr_rgb = make_rgb(
            lr_np
        )

        bicubic_rgb = make_rgb(
            bicubic_np
        )

        sr_rgb = make_rgb(
            sr_np
        )

        hr_rgb = make_rgb(
            hr_np
        )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(20, 5)
        )

        axes[0].imshow(
            lr_rgb
        )

        axes[0].set_title(
            "Sentinel-2 LR\n10 m"
        )

        axes[1].imshow(
            bicubic_rgb
        )

        axes[1].set_title(
            "Bicubic 4×\nBaseline"
        )

        axes[2].imshow(
            sr_rgb
        )

        axes[2].set_title(
            "CNN + CBAM\n4× SR"
        )

        axes[3].imshow(
            hr_rgb
        )

        axes[3].set_title(
            "NAIP HR\n2.5 m Reference"
        )

        for ax in axes:

            ax.axis("off")

        sample_name = (
            lr_path.stem
            .replace("_lr", "")
        )

        fig.suptitle(
            "SIH26142 — Super-Resolution Comparison\n"
            + sample_name,
            fontsize=14
        )

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"comparison_{image_number}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"    Saved → {output_path}"
        )


print()
print("=" * 70)
print("VISUAL COMPARISONS COMPLETE")
print("=" * 70)

print(
    f"Images saved in: {OUTPUT_DIR}"
)