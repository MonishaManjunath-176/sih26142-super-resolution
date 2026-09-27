# Deep Learning Based Super Resolution Mapping (SRM) from Medium Resolution Satellite Imageries
**SIH26142 - Smart India Hackathon Prototype**

## 📌 Project Overview
This project addresses **SIH26142**: Reconstructing high-resolution satellite imagery from medium-resolution inputs using deep learning generative models. 

Specifically, this system takes a **10m Sentinel-2 satellite image** (LR input) and performs **4× spatial upscaling** to synthesize an enhanced image matching **~2.5m NAIP resolution** (HR target).

### 🎯 Key Goals & Specifications
* **Input (LR):** 10m spatial resolution Sentinel-2 imagery (RGB + NIR channels).
* **Target (HR):** ~2.5m spatial resolution NAIP imagery.
* **Upscaling Factor:** 4× spatial resolution enhancement.
* **Dataset:** SEN2NAIP / SEN2NAIPv2 paired multi-spectral satellite imagery.
* **Architecture:** GAN framework featuring a U-Net Generator and a PatchGAN CNN Discriminator.

---

## 🏗️ Pipeline Architecture

```
                       [ Training Stage ]
                       
  10m Sentinel-2 (LR)  ──>  U-Net Generator  ──>  2.5m Generated (SR)  ──┐
                                                                         ├─> Discriminator ↔ Real 2.5m NAIP (HR)
                                                         Loss Computation ──┘ (Adversarial + L1 + Spectral Loss)


                       [ Inference Stage ]
                       
  10m Sentinel-2 (LR)  ──> Preprocessing ──> U-Net Generator ──> 4× Upscaled 2.5m SR Output Image
```

---

## 📁 Repository Directory Structure

```
sih26142_super_resolution/
├── README.md                 # Project summary, setup, pipeline, and execution commands
├── requirements.txt          # Python dependencies
├── configs/
│   └── config.yaml           # Hyperparameters, data paths, model configuration
├── data/
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset for paired Sentinel-2 (LR) & NAIP (HR) data
│   └── transforms.py         # Multi-spectral image augmentations and normalization
├── models/
│   ├── __init__.py
│   ├── generator.py          # U-Net Generator with 4x spatial upscaling blocks
│   ├── discriminator.py      # CNN PatchGAN Discriminator for high-frequency detail
│   └── loss.py               # Combined Loss (L1 Pixel + Adversarial + Spectral Consistency)
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Trainer class managing G & D updates and logging
│   └── train.py              # Main training entry point
├── inference/
│   ├── __init__.py
│   └── predict.py            # Single-image / batch inference pipeline for 10m -> 2.5m SRM
├── evaluation/
│   ├── __init__.py
│   └── metrics.py            # Evaluation metrics (PSNR, SSIM, SAM Spectral Consistency)
├── outputs/
│   ├── checkpoints/          # Model checkpoints saved during training
│   └── samples/              # Generated SR sample outputs for visual inspection
└── notebooks/
    └── EDA_and_Visualization.ipynb # Interactive notebook for data exploration and visual comparison
```

---

## 🚀 Environment & Execution Setup

### 1. Requirements & Dependencies
Ensure Python 3.8+ and PyTorch (with CUDA support) are installed along with geospatial & image processing packages:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Place paired **SEN2NAIP** dataset patches in the `data/` directory:
```
data/
├── train/
│   ├── lr/   # 10m Sentinel-2 images (e.g., 64x64)
│   └── hr/   # 2.5m NAIP images (e.g., 256x256)
└── val/
    ├── lr/
    └── hr/
```

### 3. Model Training
To launch training with default configurations:
```bash
python training/train.py --config configs/config.yaml
```

### 4. Inference / Prediction
To upscale a 10m Sentinel-2 image to 2.5m resolution:
```bash
python inference/predict.py --input path/to/sentinel2_10m.tif --output outputs/samples/sr_2.5m.tif --checkpoint outputs/checkpoints/best_generator.pth
```

### 5. Quantitative & Visual Evaluation
To evaluate PSNR, SSIM, and Spectral Consistency across validation set:
```bash
python evaluation/metrics.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_generator.pth
```

---

## 📊 Evaluation Criteria
1. **PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-level reconstruction fidelity.
2. **SSIM (Structural Similarity Index Measure):** Evaluates structural detail and edge sharpness.
3. **Spectral Consistency (SAM - Spectral Angle Mapper):** Ensures color/spectral fidelity across RGB+NIR bands.
4. **Visual Comparison:** Qualitative assessment comparing Sentinel-2 (LR), Generated (SR), and NAIP (HR).
