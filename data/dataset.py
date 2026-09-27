"""Paired Sentinel-2/NAIP dataset utilities.

Supports:
1. Local paired GeoTIFF folders
2. SEN2NAIPv2 TACO cross-sensor dataset
3. Reproducible train/validation/test splits for TACO
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError as exc:
    raise ImportError(
        "rasterio is required to train the SRM model."
    ) from exc


def paired_sample_key(path, root):
    """Return the stable identity shared by a LR/HR pair."""

    stem = Path(path).stem.lower()

    # Our extracted SEN2NAIPv2 files end with _lr / _hr
    if stem.endswith("_lr"):
        stem = stem[:-3]

    elif stem.endswith("_hr"):
        stem = stem[:-3]

    return stem


def build_pairs(lr_dir, hr_dir):
    """Build one-to-one LR/HR pairs from local GeoTIFF folders."""

    lr_root = Path(lr_dir)
    hr_root = Path(hr_dir)

    lr_files = (
        sorted(lr_root.rglob("*.tif"))
        + sorted(lr_root.rglob("*.tiff"))
    )

    hr_files = (
        sorted(hr_root.rglob("*.tif"))
        + sorted(hr_root.rglob("*.tiff"))
    )

    if not lr_files or not hr_files:
        raise FileNotFoundError(
            f"Paired data is required. "
            f"Found {len(lr_files)} LR and "
            f"{len(hr_files)} HR GeoTIFF files."
        )

    def index_files(files, root, label):

        indexed = {}

        for file_path in files:

            key = paired_sample_key(
                file_path,
                root
            )

            if key in indexed:
                raise ValueError(
                    f"Duplicate {label} sample key: {key}"
                )

            indexed[key] = file_path

        return indexed

    lr_index = index_files(
        lr_files,
        lr_root,
        "LR"
    )

    hr_index = index_files(
        hr_files,
        hr_root,
        "HR"
    )

    missing_hr = sorted(
        set(lr_index) - set(hr_index)
    )

    missing_lr = sorted(
        set(hr_index) - set(lr_index)
    )

    if missing_hr or missing_lr:

        details = []

        if missing_hr:
            details.append(
                f"missing HR for "
                f"{', '.join(missing_hr[:3])}"
            )

        if missing_lr:
            details.append(
                f"missing LR for "
                f"{', '.join(missing_lr[:3])}"
            )

        raise ValueError(
            "LR/HR files must be one-to-one pairs: "
            + "; ".join(details)
        )

    return [
        (
            key,
            lr_index[key],
            hr_index[key]
        )
        for key in sorted(lr_index)
    ]


def read_multispectral(path, in_channels=4):
    """Read multispectral bands and normalize to [0, 1]."""

    with rasterio.open(path) as src:

        if src.count < in_channels:

            raise ValueError(
                f"{path} has {src.count} bands; "
                f"{in_channels} bands are required."
            )

        data = src.read(
            list(range(1, in_channels + 1))
        ).astype(np.float32)

    max_value = float(
        np.nanmax(data)
    )

    # Sentinel-2 style reflectance
    if max_value > 255.0:

        data /= 10000.0

    # 8-bit imagery
    elif max_value > 1.0:

        data /= 255.0

    return torch.from_numpy(
        np.clip(
            data,
            0.0,
            1.0
        )
    )


class SEN2NAIPDataset(Dataset):
    """Paired RGBN dataset for 4x satellite super-resolution.

    Supports:

    Local:
        lr_dir + hr_dir

    TACO:
        SEN2NAIPv2 cross-sensor dataset

    TACO split:
        train / val / test using a saved CSV split file.
    """

    def __init__(
        self,
        lr_dir=None,
        hr_dir=None,
        in_channels=4,
        transform=None,
        is_train=True,
        scale=4,
        taco_dataset=None,
        split=None,
        split_file=None,
    ):

        self.in_channels = in_channels
        self.transform = transform
        self.scale = scale
        self.is_train = is_train

        # =========================================================
        # TACO MODE
        # =========================================================

        if taco_dataset is not None:

            self.mode = "taco"

            self.taco_dataset = taco_dataset

            # -----------------------------------------------------
            # Use saved train / validation / test split
            # -----------------------------------------------------

            if split_file is not None:

                if split not in {
                    "train",
                    "val",
                    "test"
                }:

                    raise ValueError(
                        "split must be one of: "
                        "'train', 'val', 'test'"
                    )

                try:
                    import pandas as pd

                except ImportError as exc:

                    raise ImportError(
                        "pandas is required when "
                        "using split_file."
                    ) from exc

                split_path = Path(
                    split_file
                )

                if not split_path.exists():

                    raise FileNotFoundError(
                        f"Split file not found: "
                        f"{split_file}"
                    )

                split_df = pd.read_csv(
                    split_path
                )

                required_columns = {
                    "tortilla:id",
                    "stac:centroid",
                    "split",
                }

                missing_columns = (
                    required_columns
                    - set(split_df.columns)
                )

                if missing_columns:

                    raise ValueError(
                        "Split file is missing "
                        f"columns: {missing_columns}"
                    )

                split_ids = set(
                    split_df.loc[
                        split_df["split"] == split,
                        "tortilla:id",
                    ].astype(str)
                )

                # IDs from the TACO metadata
                taco_ids = (
                    self.taco_dataset[
                        "tortilla:id"
                    ]
                    .astype(str)
                    .tolist()
                )

                # Map sample ID -> original TACO index
                taco_id_to_index = {
                    sample_id: index
                    for index, sample_id
                    in enumerate(taco_ids)
                }

                # Select original TACO indices
                self.samples = [
                    taco_id_to_index[sample_id]
                    for sample_id in split_ids
                    if sample_id in taco_id_to_index
                ]

                # Sort for reproducibility
                self.samples.sort()

                # Check that every requested split ID exists
                missing_ids = (
                    split_ids
                    - set(taco_id_to_index)
                )

                if missing_ids:

                    print(
                        "[Dataset] WARNING: "
                        f"{len(missing_ids)} IDs from "
                        f"the {split} split were not "
                        "found in the TACO dataset."
                    )

                print(
                    f"[Dataset] Using TACO "
                    f"{split} split: "
                    f"{len(self.samples)} "
                    "paired samples"
                )

            # -----------------------------------------------------
            # No split file → use all TACO samples
            # -----------------------------------------------------

            else:

                self.samples = list(
                    range(
                        len(
                            self.taco_dataset
                        )
                    )
                )

                print(
                    "[Dataset] Using full TACO "
                    f"dataset: {len(self.samples)} "
                    "paired samples"
                )

        # =========================================================
        # LOCAL MODE
        # =========================================================

        else:

            self.mode = "local"

            if (
                lr_dir is None
                or hr_dir is None
            ):

                raise ValueError(
                    "lr_dir and hr_dir are required "
                    "when taco_dataset is not provided."
                )

            self.samples = build_pairs(
                lr_dir,
                hr_dir
            )

            print(
                "[Dataset] Using local dataset: "
                f"{len(self.samples)} "
                "paired samples"
            )

    # =============================================================
    # LENGTH
    # =============================================================

    def __len__(self):

        return len(self.samples)

    # =============================================================
    # LOCAL DATASET
    # =============================================================

    def _get_local_sample(self, index):

        sample_id, lr_path, hr_path = (
            self.samples[index]
        )

        lr_img = read_multispectral(
            lr_path,
            self.in_channels
        )

        hr_img = read_multispectral(
            hr_path,
            self.in_channels
        )

        return (
            lr_img,
            hr_img,
            sample_id
        )

    # =============================================================
    # TACO DATASET
    # =============================================================

    def _get_taco_sample(self, index):

        # ---------------------------------------------------------
        # index is the index inside our split.
        #
        # self.samples[index] contains the ORIGINAL TACO index.
        # ---------------------------------------------------------

        taco_index = self.samples[index]

        # Each TACO sample contains two assets:
        #
        # read(0) -> Sentinel-2 LR
        # read(1) -> NAIP HR
        #

        sample = self.taco_dataset.read(
            taco_index
        )

        lr_path = sample.read(0)

        hr_path = sample.read(1)

        lr_img = read_multispectral(
            lr_path,
            self.in_channels
        )

        hr_img = read_multispectral(
            hr_path,
            self.in_channels
        )

        # Get the real dataset sample ID
        try:

            sample_id = str(
                self.taco_dataset.iloc[
                    taco_index
                ]["tortilla:id"]
            )

        except Exception:

            # Fallback if metadata lookup fails
            sample_id = (
                f"taco_{taco_index:05d}"
            )

        return (
            lr_img,
            hr_img,
            sample_id
        )

    # =============================================================
    # PYTORCH GETITEM
    # =============================================================

    def __getitem__(self, index):

        if self.mode == "taco":

            (
                lr_img,
                hr_img,
                sample_id
            ) = self._get_taco_sample(
                index
            )

        else:

            (
                lr_img,
                hr_img,
                sample_id
            ) = self._get_local_sample(
                index
            )

        # =========================================================
        # Verify 4x relationship
        # =========================================================

        expected_hr_size = (
            lr_img.shape[1] * self.scale,
            lr_img.shape[2] * self.scale,
        )

        if (
            tuple(hr_img.shape[1:])
            != expected_hr_size
        ):

            raise ValueError(
                f"Pair '{sample_id}' has "
                f"LR {tuple(lr_img.shape[1:])}, "
                f"HR {tuple(hr_img.shape[1:])}; "
                f"expected HR "
                f"{expected_hr_size} "
                f"for {self.scale}x SR."
            )

        # =========================================================
        # Apply transforms
        # =========================================================

        if self.transform:

            lr_img, hr_img = self.transform(
                lr_img,
                hr_img
            )

        return {
            "lr": lr_img,
            "hr": hr_img,
            "sample_id": sample_id,
        }


def load_taco_crosssensor():
    """Load the SEN2NAIPv2 cross-sensor dataset."""

    try:

        import tacoreader

    except ImportError as exc:

        raise ImportError(
            "tacoreader is required for the "
            "SEN2NAIPv2 dataset. "
            "Install it with: "
            "pip install tacoreader"
        ) from exc

    print(
        "[Dataset] Loading SEN2NAIPv2 "
        "cross-sensor dataset..."
    )

    dataset = tacoreader.load(
        "tacofoundation:sen2naipv2-crosssensor"
    )

    print(
        f"[Dataset] Loaded "
        f"{len(dataset)} samples."
    )

    return dataset