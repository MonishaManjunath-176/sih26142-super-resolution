import time

import torch
from torch.utils.data import DataLoader

from data.dataset import SEN2NAIPDataset
from data.transforms import get_transforms


LR_DIR = "data/sen2naipv2_extracted/train/lr"
HR_DIR = "data/sen2naipv2_extracted/train/hr"

SCALE = 4


def benchmark(num_workers):

    print()
    print("=" * 60)
    print(f"Testing num_workers = {num_workers}")
    print("=" * 60)

    dataset = SEN2NAIPDataset(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        in_channels=4,
        transform=get_transforms(
            True,
            SCALE
        ),
        is_train=True,
        scale=SCALE,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )

    start = time.perf_counter()

    for i, batch in enumerate(loader):

        if i >= 20:
            break

    elapsed = time.perf_counter() - start

    print(
        f"20 batches completed in "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Average: "
        f"{elapsed / 20:.3f} sec/batch"
    )


if __name__ == "__main__":

    for workers in [0, 2, 4]:

        benchmark(workers)