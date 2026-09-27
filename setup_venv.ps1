Set-Location "C:\Users\sunil\.gemini\antigravity\scratch\sih26142_super_resolution"

Write-Output "=== 1. Creating .venv virtual environment ==="
python -m venv .venv

Write-Output "=== 2. Upgrading pip inside .venv ==="
& .\.venv\Scripts\python.exe -m pip install --upgrade pip

Write-Output "=== 3. Installing CUDA-enabled PyTorch (cu121) ==="
& .\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Output "=== 4. Installing remaining project requirements ==="
& .\.venv\Scripts\pip.exe install numpy pillow scikit-image scipy matplotlib pyyaml tqdm rasterio rioxarray tifffile albumentations datasets

Write-Output "=== 5. Running PyTorch CUDA Verification ==="
& .\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if torch.cuda.is_available() else 'N/A')"
