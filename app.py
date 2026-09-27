import streamlit as st
import rasterio
import numpy as np
import torch
from pathlib import Path

from models.generator import CNNAttentionGenerator
from evaluation.metrics import calculate_psnr, calculate_ssim, calculate_sam


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SIH26142 — Satellite Super Resolution",
    page_icon="🛰️",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_PATH = OUTPUT_DIR / "app_input.tif"
SR_PATH = OUTPUT_DIR / "app_SR.tif"

# Validation data only — test set remains untouched
VAL_LR_DIR = BASE_DIR / "data" / "sen2naipv2_extracted" / "val" / "lr"
VAL_HR_DIR = BASE_DIR / "data" / "sen2naipv2_extracted" / "val" / "hr"

CHECKPOINT = OUTPUT_DIR / "checkpoints" / "best_generator.pth"


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

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
        map_location=DEVICE,
        weights_only=False
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "generator_state_dict" in checkpoint:
            state_dict = checkpoint["generator_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# IMAGE FUNCTIONS
# =========================================================

def read_reflectance(path):
    """
    Read first four bands:
    B04, B03, B02, B08

    Convert values to [0, 1].
    """

    with rasterio.open(path) as src:

        if src.count < 4:
            raise ValueError(
                "A 4-band RGBN GeoTIFF is required."
            )

        data = src.read([1, 2, 3, 4]).astype(np.float32)

    maximum = np.nanmax(data)

    if maximum > 255:
        data = data / 10000.0

    elif maximum > 1:
        data = data / 255.0

    data = np.nan_to_num(data)

    return np.clip(data, 0, 1)


def make_rgb(path):
    """
    Create a displayable RGB preview from bands 1,2,3.
    Uses percentile stretching.
    """

    with rasterio.open(path) as src:

        if src.count < 3:
            raise ValueError(
                "At least 3 bands are required for RGB preview."
            )

        data = src.read([1, 2, 3]).astype(np.float32)

    rgb = []

    for band in data:

        low = np.percentile(band, 2)
        high = np.percentile(band, 98)

        band = (band - low) / (high - low + 1e-8)

        band = np.clip(band, 0, 1)

        rgb.append(band)

    return np.stack(rgb, axis=-1)


def get_shape(path):

    with rasterio.open(path) as src:

        return (
            src.width,
            src.height,
            src.count
        )


# =========================================================
# VALIDATION PAIRS
# =========================================================

def get_validation_pairs():

    if not VAL_LR_DIR.exists() or not VAL_HR_DIR.exists():
        return []

    pairs = []

    for lr_path in sorted(VAL_LR_DIR.glob("*.tif")):

        name = lr_path.name

        if name.endswith("_lr.tif"):
            hr_name = name.replace(
                "_lr.tif",
                "_hr.tif"
            )

        else:
            hr_name = name

        hr_path = VAL_HR_DIR / hr_name

        if hr_path.exists():

            pairs.append(
                (
                    lr_path.name,
                    hr_path.name
                )
            )

    return pairs


# =========================================================
# GENERATE SUPER RESOLUTION
# =========================================================

def generate_sr(input_path, output_path):

    model = load_model()

    lr = read_reflectance(input_path)

    # [C,H,W] -> [1,C,H,W]
    lr_tensor = torch.from_numpy(lr).float()

    # Training normalization:
    # [0,1] -> [-1,1]
    lr_tensor = lr_tensor * 2.0 - 1.0

    lr_tensor = lr_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            enabled=(DEVICE.type == "cuda")
        ):

            sr = model(lr_tensor)

    # [-1,1] -> [0,1]
    sr = (sr + 1.0) / 2.0

    sr = torch.clamp(sr, 0, 1)

    sr = sr.squeeze(0).cpu().numpy()

    # -----------------------------------------------------
    # SAVE AS GEOTIFF
    # -----------------------------------------------------

    with rasterio.open(input_path) as src:

        profile = src.profile.copy()

        transform = src.transform
        crs = src.crs

        new_height = sr.shape[1]
        new_width = sr.shape[2]

        # Pixel size becomes 1/4 of original
        new_transform = transform * transform.scale(
            src.width / new_width,
            src.height / new_height
        )

        profile.update(
            height=new_height,
            width=new_width,
            count=4,
            dtype="float32",
            transform=new_transform,
            compress="deflate"
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            dst.write(sr.astype(np.float32))

    return output_path


# =========================================================
# HEADER
# =========================================================

st.title(
    "🛰️ Deep Learning Based Satellite Super Resolution"
)

st.markdown(
    """
### Enhance medium-resolution satellite imagery using AI

Our prototype uses a **CNN + CBAM Attention hybrid network**
to generate a **4× super-resolved multispectral satellite image**.

**Input:** Sentinel-2 medium-resolution imagery  
**Output:** 4× higher spatial resolution  
**Bands:** Red, Green, Blue, NIR
"""
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Model Information")

st.sidebar.write("**Architecture:** CNN + CBAM Attention")
st.sidebar.write("**Input:** 4-band RGBN")
st.sidebar.write("**Input Resolution:** 10 m")
st.sidebar.write("**Output Resolution:** 2.5 m")
st.sidebar.write("**Scale Factor:** 4×")
st.sidebar.write("**Residual Blocks:** 4")
st.sidebar.write("**Attention:** CBAM")
st.sidebar.write("**Upsampling:** PixelShuffle")
st.sidebar.write("**Loss:** L1 + Spectral Consistency")

st.sidebar.divider()

if torch.cuda.is_available():

    st.sidebar.success(
        "GPU available: CUDA"
    )

else:

    st.sidebar.warning(
        "CUDA unavailable — running on CPU."
    )

st.sidebar.write(
    f"**Device:** `{DEVICE}`"
)


# =========================================================
# CHECKPOINT CHECK
# =========================================================

if not CHECKPOINT.exists():

    st.error(
        "❌ best_generator.pth was not found."
    )

    st.stop()


# =========================================================
# VALIDATION SAMPLE SELECTION
# =========================================================

pairs = get_validation_pairs()

pair_names = [
    lr_name for lr_name, hr_name in pairs
]

st.subheader("📂 Choose Satellite Image")

selection_options = [
    "Upload your own 4-band GeoTIFF"
] + pair_names

selected = st.selectbox(
    "Select a verified validation sample or upload your own image:",
    selection_options
)


uploaded_file = None

if selected == "Upload your own 4-band GeoTIFF":

    uploaded_file = st.file_uploader(
        "Upload a 4-band RGBN GeoTIFF",
        type=["tif", "tiff"]
    )


# =========================================================
# INPUT HANDLING
# =========================================================

if selected != "Upload your own 4-band GeoTIFF":

    selected_lr = VAL_LR_DIR / selected

    # Determine HR filename
    if selected.endswith("_lr.tif"):

        hr_name = selected.replace(
            "_lr.tif",
            "_hr.tif"
        )

    else:

        hr_name = selected

    selected_hr = VAL_HR_DIR / hr_name

    # Copy LR sample to app input
    INPUT_PATH.write_bytes(
        selected_lr.read_bytes()
    )

    hr_reference_path = selected_hr

    st.success(
        f"Verified validation pair selected: {selected}"
    )


elif uploaded_file is not None:

    INPUT_PATH.write_bytes(
        uploaded_file.getbuffer()
    )

    hr_reference_path = None

else:

    st.info(
        "Choose a validation sample or upload a GeoTIFF to begin."
    )

    st.stop()


# =========================================================
# VALIDATE INPUT
# =========================================================

try:

    width, height, bands = get_shape(
        INPUT_PATH
    )

    if bands < 4:

        st.error(
            "❌ The image must contain at least 4 bands."
        )

        st.stop()

except Exception as error:

    st.error(
        f"Could not read image: {error}"
    )

    st.stop()


st.success(
    f"Image loaded — {bands} bands, "
    f"{width} × {height} pixels"
)


# =========================================================
# INPUT PREVIEW
# =========================================================

st.subheader("🔍 Input Satellite Image")

input_rgb = make_rgb(INPUT_PATH)

st.image(
    input_rgb,
    caption="Sentinel-2 LR — 10 m",
    use_container_width=True
)


st.divider()


# =========================================================
# GENERATE BUTTON
# =========================================================

if st.button(
    "🚀 Generate 4× Super-Resolution",
    use_container_width=True
):

    try:

        with st.spinner(
            "Running CNN + CBAM super-resolution model..."
        ):

            generate_sr(
                INPUT_PATH,
                SR_PATH
            )

        st.success(
            "🎉 Super-resolution generated successfully!"
        )

    except Exception as error:

        st.error(
            "❌ Model inference failed."
        )

        st.exception(error)

        st.stop()


# =========================================================
# SHOW RESULTS
# =========================================================

if SR_PATH.exists():

    sr_width, sr_height, sr_bands = get_shape(
        SR_PATH
    )

    st.subheader(
        "✨ Super-Resolved Output"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Input",
            f"{width} × {height}"
        )

    with col2:

        st.metric(
            "Output",
            f"{sr_width} × {sr_height}"
        )

    with col3:

        st.metric(
            "Scale",
            "4×"
        )


    # -----------------------------------------------------
    # SR IMAGE
    # -----------------------------------------------------

    sr_rgb = make_rgb(
        SR_PATH
    )

    st.image(
        sr_rgb,
        caption="CNN + CBAM — 4× Super-Resolution",
        use_container_width=True
    )


    # =====================================================
    # LR / SR / HR COMPARISON
    # =====================================================

    if hr_reference_path is not None:

        st.divider()

        st.subheader(
            "📊 LR vs SR vs HR Reference"
        )

        hr_rgb = make_rgb(
            hr_reference_path
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(
                input_rgb,
                caption="LR — Sentinel-2 10 m",
                use_container_width=True
            )

        with col2:

            st.image(
                sr_rgb,
                caption="SR — CNN + CBAM 4×",
                use_container_width=True
            )

        with col3:

            st.image(
                hr_rgb,
                caption="HR — NAIP 2.5 m Reference",
                use_container_width=True
            )


        # =================================================
        # METRICS
        # =================================================

        st.subheader(
            "📈 Image Quality Metrics"
        )

        try:

            sr_reflectance = read_reflectance(
                SR_PATH
            )

            hr_reflectance = read_reflectance(
                hr_reference_path
            )

            if sr_reflectance.shape != hr_reflectance.shape:

                st.warning(
                    "SR and HR dimensions do not match. "
                    "Metrics cannot be calculated."
                )

            else:

                psnr = calculate_psnr(
                    sr_reflectance,
                    hr_reflectance
                )

                ssim = calculate_ssim(
                    sr_reflectance,
                    hr_reflectance
                )

                sam = calculate_sam(
                    sr_reflectance,
                    hr_reflectance
                )

                metric1, metric2, metric3 = st.columns(3)

                with metric1:

                    st.metric(
                        "PSNR",
                        f"{psnr:.2f} dB"
                    )

                with metric2:

                    st.metric(
                        "SSIM",
                        f"{ssim:.4f}"
                    )

                with metric3:

                    st.metric(
                        "SAM",
                        f"{sam:.2f}°"
                    )

        except Exception as error:

            st.warning(
                f"Metrics could not be calculated: {error}"
            )


# =========================================================
# BICUBIC BASELINE — REFERENCE VALUES
# =========================================================

st.divider()

st.subheader(
    "📊 Test Set Performance"
)

st.markdown(
    """
The following results were calculated on the **816-sample
held-out test set**.
"""
)

metric1, metric2, metric3 = st.columns(3)

with metric1:

    st.metric(
        "CNN + CBAM PSNR",
        "35.7275 dB"
    )

with metric2:

    st.metric(
        "CNN + CBAM SSIM",
        "0.8697"
    )

with metric3:

    st.metric(
        "CNN + CBAM SAM",
        "2.3608°"
    )


st.caption(
    "Test set: 816 paired Sentinel-2 / NAIP samples."
)


# =========================================================
# DOWNLOAD
# =========================================================

if SR_PATH.exists():

    st.divider()

    st.subheader(
        "📥 Download Result"
    )

    with open(
        SR_PATH,
        "rb"
    ) as file:

        st.download_button(
            label="📥 Download Super-Resolved GeoTIFF",
            data=file,
            file_name="SIH26142_CNN_CBAM_SR.tif",
            mime="image/tiff",
            use_container_width=True
        )


# =========================================================
# TECHNICAL DETAILS
# =========================================================

st.divider()

st.subheader(
    "🔬 Technical Details"
)

st.markdown(
    """
### Processing Pipeline

**Sentinel-2 LR GeoTIFF**  
↓  
**Preprocessing / Normalization**  
↓  
**Residual CNN Blocks**  
↓  
**CBAM Attention**
- Channel Attention
- Spatial Attention  
↓  
**PixelShuffle ×2**  
↓  
**PixelShuffle ×2**  
↓  
**4× Super-Resolved RGBN Image**

### Model

- Input: 4-channel RGBN
- Input spatial resolution: 10 m
- Output spatial resolution: 2.5 m
- Upscaling factor: 4×
- Residual CNN blocks: 4
- Attention: CBAM
- Upsampling: PixelShuffle
- Reconstruction loss: L1
- Spectral loss: Spectral consistency / SAM
- Discriminator: None

### Evaluation

The model was evaluated using:

- **PSNR** — reconstruction fidelity
- **SSIM** — structural similarity
- **SAM** — spectral similarity

The held-out test set contains **816 paired samples**.

The current prototype implements the **super-resolution stage**.
The downstream **super-resolution mapping / fraction estimation**
stage is planned as the next extension.
"""
)