import shutil
import os
import sys

print("=== STEP 1: DISK SPACE CHECK ===")
total, used, free = shutil.disk_usage("C:\\")
print(f"C: Drive Total: {total / (1024**3):.2f} GB")
print(f"C: Drive Used:  {used / (1024**3):.2f} GB")
print(f"C: Drive Free:  {free / (1024**3):.2f} GB")

print("\n=== STEP 2: INSPECTING SEN2NAIPv2 DATASET ACCESS ===")
try:
    from datasets import load_dataset
    print("HuggingFace 'datasets' library is imported successfully.")
    
    # Try streaming dataset or downloading only 2 samples
    print("Attempting to load a tiny 2-sample subset from HuggingFace...")
    ds = load_dataset("tacofoundation/SEN2NAIPv2", split="train[:2]", streaming=False)
    print("Successfully loaded dataset split! Features:", ds.features)
    print("Number of samples in test subset:", len(ds))
    
    sample = ds[0]
    print("Keys in sample:", sample.keys())
    
except Exception as e:
    print("Error during HuggingFace dataset loading:", e)
