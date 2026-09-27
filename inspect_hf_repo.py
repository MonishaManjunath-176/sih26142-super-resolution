import shutil
from huggingface_hub import HfApi, hf_hub_download

print("=== STEP 1: DISK SPACE ===")
total, used, free = shutil.disk_usage("C:\\")
print(f"C: Drive Free Space: {free / (1024**3):.2f} GB (Total: {total / (1024**3):.2f} GB)")

print("\n=== STEP 2: REPO FILES INSPECTION ===")
api = HfApi()

for repo in ["isp-uv-es/SEN2NAIP", "tacofoundation/SEN2NAIPv2"]:
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        print(f"\nRepo: {repo} (Total Files: {len(files)})")
        for f in files[:15]:
            print(f"  - {f}")
    except Exception as e:
        print(f"Error inspecting {repo}: {e}")
