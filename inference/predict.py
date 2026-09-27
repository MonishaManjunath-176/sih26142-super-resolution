import argparse
import os

import numpy as np
import rasterio
import torch

from models.generator import CNNAttentionGenerator


def load_tiff(input_path):
    """Load a multispectral GeoTIFF using Rasterio."""

    with rasterio.open(input_path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()

    # Ensure exactly four channels: RGBN.
    if data.shape[0] < 4:
        pad_channels = 4 - data.shape[0]
        data = np.concatenate(
            [
                data,
                np.repeat(
                    data[-1:],
                    pad_channels,
                    axis=0
                ),
            ],
            axis=0,
        )
    elif data.shape[0] > 4:
        data = data[:4]

    # Same normalization convention used by the dataset:
    # reflectance-like values -> [0, 1].
    max_val = data.max()

    if max_val > 255.0:
        data = data / 10000.0
    elif max_val > 1.0:
        data = data / 255.0

    data = np.clip(data, 0.0, 1.0)

    return data, profile


def run_inference(
    input_path,
    output_path,
    checkpoint_path,
    in_channels=4,
    base_features=64,
    upscale_factor=4,
    num_res_blocks=4,
    attention_reduction=8,
):
    """
    Sentinel-2-style 4-band multispectral TIFF
        |
        v
    CNN feature extraction
        |
        v
    Residual CNN blocks
        |
        v
    CBAM channel + spatial attention
        |
        v
    PixelShuffle x2 -> PixelShuffle x2
        |
        v
    4-band 4x super-resolved GeoTIFF
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"[Info] Running inference on device: {device}"
    )

    # --------------------------------------------------
    # Build CNN + Attention Generator
    # --------------------------------------------------
    generator = CNNAttentionGenerator(
        in_channels=in_channels,
        out_channels=in_channels,
        base_features=base_features,
        upscale_factor=upscale_factor,
        num_res_blocks=num_res_blocks,
        attention_reduction=attention_reduction,
    ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    generator.load_state_dict(state_dict)
    generator.eval()

    print(
        f"[Info] Successfully loaded checkpoint from "
        f"{checkpoint_path}"
    )

    # --------------------------------------------------
    # Load multispectral input
    # --------------------------------------------------
    data, profile = load_tiff(input_path)

    print(
        f"[Info] Input shape: {data.shape}"
    )

    img_tensor = torch.from_numpy(data).unsqueeze(0)

    # Match training normalization: [0, 1] -> [-1, 1].
    input_tensor = (
        img_tensor * 2.0 - 1.0
    ).to(device)

    # --------------------------------------------------
    # Generate super-resolved image
    # --------------------------------------------------
    with torch.no_grad():
        sr_tensor = generator(input_tensor)

    # Convert [-1, 1] -> [0, 1].
    sr_tensor = (
        sr_tensor.squeeze(0).cpu() + 1.0
    ) / 2.0

    sr_tensor = torch.clamp(
        sr_tensor,
        0.0,
        1.0
    )

    sr_np = sr_tensor.numpy().astype(
        np.float32
    )

    print(
        f"[Info] Output shape: {sr_np.shape}"
    )

    # --------------------------------------------------
    # Save as 4-band GeoTIFF
    # --------------------------------------------------
    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True
    )

    profile.update(
        driver="GTiff",
        height=sr_np.shape[1],
        width=sr_np.shape[2],
        count=4,
        dtype="float32",
    )

    # If georeferencing exists, reduce pixel size by 4x.
    if profile.get("transform") is not None:
        transform = profile["transform"]
        profile["transform"] = (
            transform * transform.scale(
                0.25,
                0.25
            )
        )

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:
        dst.write(sr_np)

    print(
        "[Success] 4x CNN + Attention "
        f"Super-Resolution GeoTIFF saved to: "
        f"{output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "CNN + CBAM Attention based "
            "4x multispectral satellite "
            "Super-Resolution"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input 4-band multispectral GeoTIFF",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output 4-band super-resolved GeoTIFF",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Trained CNN + Attention checkpoint",
    )

    parser.add_argument(
        "--base-features",
        type=int,
        default=64,
        help="Number of CNN feature channels",
    )

    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=4,
        help="Number of residual CNN blocks",
    )

    parser.add_argument(
        "--attention-reduction",
        type=int,
        default=8,
        help="CBAM channel-attention reduction ratio",
    )

    args = parser.parse_args()

    run_inference(
        input_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        base_features=args.base_features,
        num_res_blocks=args.num_res_blocks,
        attention_reduction=args.attention_reduction,
    )


if __name__ == "__main__":
    main()
