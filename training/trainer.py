import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SEN2NAIPDataset
from data.transforms import get_transforms

from evaluation.metrics import (
    calculate_psnr,
    calculate_sam,
    calculate_ssim,
)

from models.generator import CNNAttentionGenerator
from models.loss import SuperResolutionLoss


class SRMTrainingPipeline:
    """
    Train the CNN + CBAM Attention super-resolution model
    on the local SEN2NAIPv2 cross-sensor paired dataset.

    Current implementation covers the Super-Resolution
    stage of the overall SRM pipeline.

    Dataset:
        SEN2NAIPv2 cross-sensor

    Split:
        Train -> 6377 samples
        Val   -> 807 samples
        Test  -> 816 samples

    The test set is NOT used during training.
    """

    def __init__(self, config, max_batches=None):

        self.config = config
        self.max_batches = max_batches

        training = config["training"]
        data = config["data"]
        model_cfg = config["model"]["generator"]

        # =========================================================
        # DEVICE
        # =========================================================

        self.device = torch.device(
            training["device"]
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[Info] Training device: {self.device}")

        # =========================================================
        # BASIC SETTINGS
        # =========================================================

        scale = config["project"]["upscale_factor"]

        # Local extracted dataset
        dataset_root = data.get(
            "local_dataset_root",
            "data/sen2naipv2_extracted"
        )

        train_lr_dir = os.path.join(
            dataset_root,
            "train",
            "lr"
        )

        train_hr_dir = os.path.join(
            dataset_root,
            "train",
            "hr"
        )

        val_lr_dir = os.path.join(
            dataset_root,
            "val",
            "lr"
        )

        val_hr_dir = os.path.join(
            dataset_root,
            "val",
            "hr"
        )

        # =========================================================
        # CNN + CBAM GENERATOR
        # =========================================================

        self.net_g = CNNAttentionGenerator(
            in_channels=data["in_channels"],
            out_channels=data["out_channels"],
            base_features=model_cfg["base_features"],
            upscale_factor=scale,
            num_res_blocks=model_cfg["num_res_blocks"],
            attention_reduction=model_cfg[
                "attention_reduction"
            ],
        ).to(self.device)

        # =========================================================
        # OPTIMIZER
        # =========================================================

        self.opt_g = torch.optim.Adam(
            self.net_g.parameters(),
            lr=training["lr_g"],
            betas=(
                training["beta1"],
                training["beta2"],
            ),
        )

        self.scheduler_g = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.opt_g,
                mode="min",
                factor=0.5,
                patience=3,
            )
        )

        # =========================================================
        # LOSS
        # =========================================================

        self.criterion = SuperResolutionLoss(
            lambda_l1=training["lambda_l1"],
            lambda_spectral=training["lambda_spectral"],
        ).to(self.device)

        # =========================================================
        # MIXED PRECISION
        # =========================================================

        self.use_amp = (
            training.get("amp", True)
            and self.device.type == "cuda"
        )

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        # =========================================================
        # TRAIN DATASET
        # =========================================================

        print("[Info] Creating local training dataset...")

        train_dataset = SEN2NAIPDataset(
            lr_dir=train_lr_dir,
            hr_dir=train_hr_dir,
            in_channels=data["in_channels"],
            transform=get_transforms(
                True,
                scale
            ),
            is_train=True,
            scale=scale,
        )

        # =========================================================
        # VALIDATION DATASET
        # =========================================================

        print("[Info] Creating local validation dataset...")

        val_dataset = SEN2NAIPDataset(
            lr_dir=val_lr_dir,
            hr_dir=val_hr_dir,
            in_channels=data["in_channels"],
            transform=get_transforms(
                False,
                scale
            ),
            is_train=False,
            scale=scale,
        )

        # =========================================================
        # TRAIN DATALOADER
        # =========================================================

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=training["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=(
                self.device.type == "cuda"
            ),
        )

        # =========================================================
        # VALIDATION DATALOADER
        # =========================================================

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                self.device.type == "cuda"
            ),
        )

        print(
            f"[Info] Training samples: "
            f"{len(train_dataset)}"
        )

        print(
            f"[Info] Validation samples: "
            f"{len(val_dataset)}"
        )

        print(
            "[Info] Test samples: 816 "
            "(kept untouched)"
        )

        # =========================================================
        # CHECKPOINT SETUP
        # =========================================================

        self.checkpoint_dir = (
            training["checkpoint_dir"]
        )

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True
        )

        self.best_validation_loss = float("inf")

    # =============================================================
    # TRAIN ONE EPOCH
    # =============================================================

    def train_epoch(self, epoch):

        self.net_g.train()

        totals = {
            "loss": 0.0,
            "l1": 0.0,
            "spectral": 0.0,
        }

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}",
        )

        batches_processed = 0

        for batch in progress:

            # -----------------------------------------------------
            # Optional benchmark limit
            # -----------------------------------------------------

            if (
                self.max_batches is not None
                and batches_processed >= self.max_batches
            ):
                break

            lr = batch["lr"].to(
                self.device,
                non_blocking=True,
            )

            hr = batch["hr"].to(
                self.device,
                non_blocking=True,
            )

            self.opt_g.zero_grad(
                set_to_none=True
            )

            # -----------------------------------------------------
            # Forward pass
            # -----------------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.use_amp,
            ):

                sr = self.net_g(lr)

                loss, loss_items = (
                    self.criterion(sr, hr)
                )

            # -----------------------------------------------------
            # Backpropagation
            # -----------------------------------------------------

            self.scaler.scale(
                loss
            ).backward()

            self.scaler.step(
                self.opt_g
            )

            self.scaler.update()

            # -----------------------------------------------------
            # Accumulate metrics
            # -----------------------------------------------------

            totals["loss"] += loss.item()

            totals["l1"] += (
                loss_items["l1_loss"]
            )

            totals["spectral"] += (
                loss_items["spectral_loss"]
            )

            batches_processed += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                l1=f"{loss_items['l1_loss']:.4f}",
                sam=(
                    f"{loss_items['spectral_loss']:.4f}"
                ),
            )

        n = max(batches_processed, 1)

        return {
            key: value / n
            for key, value in totals.items()
        }

    # =============================================================
    # VALIDATION
    # =============================================================

    @torch.no_grad()
    def validate(self):

        self.net_g.eval()

        totals = {
            "loss": 0.0,
            "l1": 0.0,
            "spectral": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "sam": 0.0,
        }

        progress = tqdm(
            self.val_loader,
            desc="Validation",
        )

        for batch in progress:

            lr = batch["lr"].to(
                self.device,
                non_blocking=True,
            )

            hr = batch["hr"].to(
                self.device,
                non_blocking=True,
            )

            # -----------------------------------------------------
            # Forward pass
            # -----------------------------------------------------

            with torch.amp.autocast(
                device_type="cuda",
                enabled=self.use_amp,
            ):

                sr = self.net_g(lr)

                loss, loss_items = (
                    self.criterion(sr, hr)
                )

            # -----------------------------------------------------
            # Loss metrics
            # -----------------------------------------------------

            totals["loss"] += loss.item()

            totals["l1"] += (
                loss_items["l1_loss"]
            )

            totals["spectral"] += (
                loss_items["spectral_loss"]
            )

            # -----------------------------------------------------
            # Image quality metrics
            # -----------------------------------------------------

            sr_cpu = (
                sr.squeeze(0)
                .float()
                .cpu()
            )

            hr_cpu = (
                hr.squeeze(0)
                .float()
                .cpu()
            )

            totals["psnr"] += calculate_psnr(
                sr_cpu,
                hr_cpu
            )

            totals["ssim"] += calculate_ssim(
                sr_cpu,
                hr_cpu
            )

            totals["sam"] += calculate_sam(
                sr_cpu,
                hr_cpu
            )

        n = max(
            len(self.val_loader),
            1
        )

        return {
            key: value / n
            for key, value in totals.items()
        }

    # =============================================================
    # SAVE CHECKPOINT
    # =============================================================

    def save_checkpoint(self, filename):

        path = os.path.join(
            self.checkpoint_dir,
            filename
        )

        torch.save(
            self.net_g.state_dict(),
            path
        )

        return path

    # =============================================================
    # TRAINING LOOP
    # =============================================================

    def run(self):

        epochs = self.config[
            "training"
        ]["epochs"]

        print()
        print("=" * 60)
        print(
            "CNN + CBAM SATELLITE "
            "SUPER-RESOLUTION TRAINING"
        )
        print("=" * 60)

        print(
            f"[Info] Device: {self.device}"
        )

        print(
            f"[Info] AMP enabled: {self.use_amp}"
        )

        print(
            "[Info] Model: "
            "CNN + CBAM Attention"
        )

        print(
            "[Info] GAN discriminator: "
            "disabled"
        )

        print(
            "[Info] Upscaling: 4x"
        )

        print(
            f"[Info] Epochs: {epochs}"
        )

        print(
            f"[Info] Train samples: "
            f"{len(self.train_loader.dataset)}"
        )

        print(
            f"[Info] Validation samples: "
            f"{len(self.val_loader.dataset)}"
        )

        print(
            "[Info] Test set: 816 samples "
            "(not used during training)"
        )

        if self.max_batches is not None:
            print(
                f"[Info] Benchmark limit: "
                f"{self.max_batches} training batches"
            )

        print("=" * 60)
        print()

        # =========================================================
        # EPOCH LOOP
        # =========================================================

        for epoch in range(epochs):

            train = self.train_epoch(
                epoch
            )

            # -----------------------------------------------------
            # IMPORTANT:
            # During benchmark mode we skip full validation.
            # -----------------------------------------------------

            if self.max_batches is not None:

                print()
                print(
                    f"Benchmark completed after "
                    f"{self.max_batches} training batches."
                )

                print(
                    f"Training loss: "
                    f"{train['loss']:.4f}"
                )

                print(
                    f"Training L1: "
                    f"{train['l1']:.4f}"
                )

                print(
                    f"Training spectral loss: "
                    f"{train['spectral']:.4f}"
                )

                print()
                print(
                    "Full validation was skipped "
                    "during benchmark."
                )

                return

            # -----------------------------------------------------
            # Normal validation
            # -----------------------------------------------------

            metrics = self.validate()

            # -----------------------------------------------------
            # Learning-rate scheduler
            # -----------------------------------------------------

            self.scheduler_g.step(
                metrics["loss"]
            )

            # -----------------------------------------------------
            # Print results
            # -----------------------------------------------------

            print()

            print(
                f"Epoch {epoch + 1}: "
                f"Loss={train['loss']:.4f} "
                f"L1={train['l1']:.4f} "
                f"TrainSAM={train['spectral']:.4f} | "
                f"ValLoss={metrics['loss']:.4f} "
                f"PSNR={metrics['psnr']:.2f} "
                f"SSIM={metrics['ssim']:.4f} "
                f"SAM={metrics['sam']:.2f}°"
            )

            # -----------------------------------------------------
            # Save best model
            # -----------------------------------------------------

            if (
                metrics["loss"]
                < self.best_validation_loss
            ):

                self.best_validation_loss = (
                    metrics["loss"]
                )

                path = self.save_checkpoint(
                    "best_generator.pth"
                )

                print(
                    f"[Info] Saved best model: "
                    f"{path}"
                )

            # -----------------------------------------------------
            # Save regular checkpoint
            # -----------------------------------------------------

            if (
                (epoch + 1)
                % self.config["training"]["save_every"]
                == 0
            ):

                path = self.save_checkpoint(
                    f"generator_epoch_{epoch + 1}.pth"
                )

                print(
                    f"[Info] Saved checkpoint: "
                    f"{path}"
                )

        print()
        print("=" * 60)
        print("Training completed.")
        print(
            f"Best validation loss: "
            f"{self.best_validation_loss:.6f}"
        )
        print("=" * 60)