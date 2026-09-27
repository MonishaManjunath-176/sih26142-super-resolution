@echo off
echo === CHECK 1: nvidia-smi ===
nvidia-smi
echo.
echo === CHECK 2: where python ===
where python
echo.
echo === CHECK 3: python --version ===
python --version
echo.
echo === CHECK 4: PyTorch CUDA Check ===
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
