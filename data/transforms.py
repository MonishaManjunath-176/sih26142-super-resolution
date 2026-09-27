import torch
import random


class PairedRandomCrop:
    """Random crop on paired LR and HR images maintaining 4x spatial scale."""
    def __init__(self, lr_crop_size=64, scale=4):
        self.lr_crop_size = lr_crop_size
        self.hr_crop_size = lr_crop_size * scale
        self.scale = scale

    def __call__(self, lr_tensor, hr_tensor):
        _, h_lr, w_lr = lr_tensor.shape
        if h_lr > self.lr_crop_size and w_lr > self.lr_crop_size:
            top_lr = random.randint(0, h_lr - self.lr_crop_size)
            left_lr = random.randint(0, w_lr - self.lr_crop_size)

            top_hr = top_lr * self.scale
            left_hr = left_lr * self.scale

            lr_crop = lr_tensor[:, top_lr:top_lr + self.lr_crop_size, left_lr:left_lr + self.lr_crop_size]
            hr_crop = hr_tensor[:, top_hr:top_hr + self.hr_crop_size, left_hr:left_hr + self.hr_crop_size]
            return lr_crop, hr_crop
        return lr_tensor, hr_tensor


class PairedTransforms:
    def __init__(self, is_train=True, scale=4):
        self.is_train = is_train
        self.scale = scale

    def __call__(self, lr_img, hr_img):
        # Normalize range to [-1, 1] for GAN training
        lr_norm = lr_img * 2.0 - 1.0
        hr_norm = hr_img * 2.0 - 1.0

        if self.is_train:
            # Synchronized Horizontal Flip
            if random.random() > 0.5:
                lr_norm = torch.flip(lr_norm, dims=[2])
                hr_norm = torch.flip(hr_norm, dims=[2])
            # Synchronized Vertical Flip
            if random.random() > 0.5:
                lr_norm = torch.flip(lr_norm, dims=[1])
                hr_norm = torch.flip(hr_norm, dims=[1])

        return lr_norm, hr_norm


def get_transforms(is_train=True, scale=4):
    return PairedTransforms(is_train=is_train, scale=scale)
