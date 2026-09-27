Write-Output "=== CHECK 1: nvidia-smi ==="
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi
} else {
    Write-Output "'nvidia-smi' is not recognized as an internal or external command, operable program or batch file."
}

Write-Output ""
Write-Output "=== CHECK 2: where python ==="
$whereOut = cmd /c "where python 2>&1"
if ($whereOut) {
    Write-Output $whereOut
} else {
    Write-Output "INFO: Could not find files for the given pattern(s)."
}

Write-Output ""
Write-Output "=== CHECK 3: python --version ==="
$verOut = cmd /c "python --version 2>&1"
if ($verOut) {
    Write-Output $verOut
} else {
    Write-Output "Python was not found / App Execution Alias active."
}

Write-Output ""
Write-Output "=== CHECK 4: PyTorch CUDA Check ==="
$pyCode = "import sys; print('Python executable:', sys.executable); import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
$torchOut = cmd /c "python -c ""$pyCode"" 2>&1"
if ($torchOut) {
    Write-Output $torchOut
} else {
    Write-Output "Could not execute Python PyTorch snippet."
}
