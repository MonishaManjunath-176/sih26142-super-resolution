import sys

print("--- ENVIRONMENT CHECK ---")
print("Python:", sys.version)

try:
    import torch
    print("PyTorch Version:", torch.__version__)
    cuda_avail = torch.cuda.is_available()
    print("CUDA Available:", cuda_avail)
    if cuda_avail:
        print("CUDA Version (Torch):", torch.version.cuda)
        print("Device Count:", torch.cuda.device_count())
        print("Device Name:", torch.cuda.get_device_name(0))
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM: {vram_bytes / (1024**3):.2f} GB ({vram_bytes} bytes)")
    else:
        print("CUDA is NOT available in this PyTorch installation.")
except Exception as e:
    print("Error during PyTorch check:", e)
