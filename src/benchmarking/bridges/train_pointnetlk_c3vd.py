#!/usr/bin/env python
# ruff: noqa: E402,I001
"""
Training script for PointNetLK on C3VD dataset.

This script supports two-stage training:
1. Stage 1: Train classifier (feature extractor)
2. Stage 2: Train PointNetLK (registration) with transfer learning

Usage:
    # Stage 1: Train classifier
    python src/benchmarking/bridges/train_pointnetlk_c3vd.py \\
        --config src/benchmarking/bridges/configs/c3vd_pointnetlk.yaml \\
        --stage classifier \\
        --data-root /path/to/C3VD_datasets

    # Stage 2: Train PointNetLK
    python src/benchmarking/bridges/train_pointnetlk_c3vd.py \\
        --config src/benchmarking/bridges/configs/c3vd_pointnetlk.yaml \\
        --stage pointnetlk \\
        --data-root /path/to/C3VD_datasets \\
        --transfer-from experiments/checkpoints/c3vd_pointnetlk/classifier_feat_best.pth
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import argparse
import torch
import torch.utils.data
import yaml

# Add PointNetLK to path
BRIDGE_DIR = Path(__file__).resolve().parent
SRC_ROOT = BRIDGE_DIR.parents[1]
REPO_ROOT = SRC_ROOT.parent
pointnetlk_path = REPO_ROOT / "baselines" / "PointNetLK"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(pointnetlk_path))
import ptlk

from common.datasets.c3vd_for_pointnetlk import C3VDForPointNetLK, C3VDset4tracking

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _batch_size_from_points(points):
    if isinstance(points, (list, tuple)):
        points = points[0]
    return points.size(0)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_output_dir(config, stage):
    """Setup output directory structure."""
    output_dir = Path(config["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["output"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = log_dir / f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Log file: {log_file}")

    return output_dir, log_dir


def get_datasets(config, data_root):
    """Create train and test datasets."""
    logger.info("Creating datasets...")
    dataset_kwargs = {
        "train_scenes": config["dataset"].get("train_scenes"),
        "val_scenes": config["dataset"].get("val_scenes"),
        "test_scenes": config["dataset"].get("test_scenes"),
        "frame_stride": config["dataset"].get("frame_stride", 1),
        "sampling_mode": config["dataset"].get("sampling_mode", "voxel"),
        "normalize_mode": config["dataset"].get("normalize_mode", "unit_cube"),
        "classifier_label_mode": config["dataset"].get(
            "classifier_label_mode",
            "scene",
        ),
    }

    # Create base datasets
    trainset = C3VDForPointNetLK(
        data_root=data_root,
        num_points=config["dataset"]["num_points"],
        split="train",
        train_ratio=config["dataset"]["train_ratio"],
        random_seed=config["dataset"]["random_seed"],
        max_pairs=config["dataset"].get("max_train_pairs"),
        **dataset_kwargs,
    )

    valset = C3VDForPointNetLK(
        data_root=data_root,
        num_points=config["dataset"]["num_points"],
        split="val",
        train_ratio=config["dataset"]["train_ratio"],
        random_seed=config["dataset"]["random_seed"],
        max_pairs=config["dataset"].get("max_val_pairs")
        or config["dataset"].get("max_test_pairs"),
        **dataset_kwargs,
    )

    logger.info(f"Train set size: {len(trainset)}")
    logger.info(f"Val set size: {len(valset)}")
    logger.info(
        "Classifier classes "
        f"({dataset_kwargs['classifier_label_mode']}): {trainset.classes}"
    )
    logger.info(f"Number of classifier classes: {trainset.num_classes}")

    return trainset, valset


def save_checkpoint(state, filepath):
    """Save checkpoint to file."""
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _loss_is_usable(loss, max_loss=None):
    if not torch.isfinite(loss):
        return False
    if max_loss is not None and float(loss.detach().item()) > float(max_loss):
        return False
    return True


# ==================== Classifier Training ====================


class ClassifierTrainer:
    """Trainer for PointNet classifier (Stage 1)."""

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device(config["training"]["device"])
        classifier_cfg = config["training"]["classifier"]
        self.max_train_steps = classifier_cfg.get("max_train_steps")
        self.max_test_steps = classifier_cfg.get("max_test_steps")
        self.grad_clip_norm = classifier_cfg.get("grad_clip_norm")

        # Model parameters
        self.num_classes = None  # Set when creating model
        self.dim_k = config["model"]["dim_k"]
        self.use_tnet = config["model"]["use_tnet"]

        # Symmetric function
        symfn_name = config["model"]["symfn"]
        if symfn_name == "max":
            self.sym_fn = ptlk.pointnet.symfn_max
        elif symfn_name == "avg":
            self.sym_fn = ptlk.pointnet.symfn_avg
        else:
            raise ValueError(f"Unknown symfn: {symfn_name}")

    def create_model(self, num_classes):
        """Create PointNet classifier model."""
        self.num_classes = num_classes
        feat = ptlk.pointnet.PointNet_features(self.dim_k, self.use_tnet, self.sym_fn)
        model = ptlk.pointnet.PointNet_classifier(self.num_classes, feat, self.dim_k)
        logger.info("Created PointNet classifier:")
        logger.info(f"  Dim K: {self.dim_k}")
        logger.info(f"  Use TNet: {self.use_tnet}")
        logger.info(f"  Symmetric function: {self.config['model']['symfn']}")
        logger.info(f"  Number of classes: {self.num_classes}")
        return model

    def compute_loss(self, model, data, device):
        """Compute classification loss."""
        points, target = data
        if isinstance(points, (list, tuple)):
            # The C3VD adapter returns (source, target) so classifier training
            # needs to pick one concrete point cloud tensor.
            points = points[1]
        points = points.to(device)
        target = target.to(device)

        output = model(points)
        loss = torch.nn.functional.cross_entropy(output, target)

        return target, output, loss

    def train_epoch(self, model, dataloader, optimizer, device):
        """Train one epoch."""
        model.train()
        total_loss = 0.0
        correct = 0
        count = 0

        for batch_idx, data in enumerate(dataloader):
            if self.max_train_steps is not None and batch_idx >= self.max_train_steps:
                break
            target, output, loss = self.compute_loss(model, data, device)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.grad_clip_norm,
                )
            optimizer.step()

            # Statistics
            total_loss += loss.item() * output.size(0)
            count += output.size(0)

            _, pred = output.max(dim=1)
            correct += (pred == target).sum().item()

            if batch_idx % self.config["logging"]["log_interval"] == 0:
                logger.info(
                    f"  Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Acc: {100.0 * correct / count:.2f}%"
                )

        avg_loss = total_loss / count
        accuracy = 100.0 * correct / count
        return avg_loss, accuracy

    def eval_epoch(self, model, dataloader, device):
        """Evaluate one epoch."""
        model.eval()
        total_loss = 0.0
        correct = 0
        count = 0

        with torch.no_grad():
            for data in dataloader:
                if (
                    self.max_test_steps is not None
                    and count > 0
                    and count >= self.max_test_steps * _batch_size_from_points(data[0])
                ):
                    break
                target, output, loss = self.compute_loss(model, data, device)

                total_loss += loss.item() * output.size(0)
                count += output.size(0)

                _, pred = output.max(dim=1)
                correct += (pred == target).sum().item()

        avg_loss = total_loss / count
        accuracy = 100.0 * correct / count
        return avg_loss, accuracy

    def train(self, trainset, testset, epochs, dev_mode=False):
        """Main training loop for classifier."""
        logger.info("=" * 60)
        logger.info("Starting Classifier Training (Stage 1)")
        logger.info("=" * 60)

        if dev_mode:
            epochs = min(epochs, 3)
            logger.info(f"DEV MODE: Reducing epochs to {epochs}")

        # Create model
        model = self.create_model(trainset.num_classes)
        model.to(self.device)

        # Create dataloaders
        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"]["num_workers"]

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Setup optimizer
        classifier_cfg = self.config["training"]["classifier"]
        learning_rate = float(classifier_cfg.get("learning_rate", 1.0e-3))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        logger.info(f"Classifier optimizer: Adam(lr={learning_rate:.2e})")
        if self.grad_clip_norm is not None:
            logger.info(f"Classifier gradient clipping: {self.grad_clip_norm}")

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=float(classifier_cfg.get("min_lr", 1.0e-7)),
        )
        logger.info("✓ Using ReduceLROnPlateau scheduler (factor=0.5, patience=5)")

        # Training loop
        best_loss = float("inf")
        best_acc = 0.0

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss, train_acc = self.train_epoch(
                model, trainloader, optimizer, self.device
            )
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            # Evaluate
            val_loss, val_acc = self.eval_epoch(model, testloader, self.device)
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Learning Rate: {current_lr:.2e}")

            # Save checkpoints
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            best_acc = max(best_acc, val_acc)

            # Save model
            prefix = self.config["output"]["classifier_prefix"]
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "min_loss": best_loss,
                "optimizer": optimizer.state_dict(),
            }

            # Always save last
            save_checkpoint(checkpoint, self.output_dir / f"{prefix}_snap_last.pth")
            save_checkpoint(
                model.state_dict(), self.output_dir / f"{prefix}_model_last.pth"
            )
            save_checkpoint(
                model.features.state_dict(), self.output_dir / f"{prefix}_feat_last.pth"
            )

            if is_best:
                save_checkpoint(checkpoint, self.output_dir / f"{prefix}_snap_best.pth")
                save_checkpoint(
                    model.state_dict(), self.output_dir / f"{prefix}_model_best.pth"
                )
                save_checkpoint(
                    model.features.state_dict(),
                    self.output_dir / f"{prefix}_feat_best.pth",
                )
                logger.info(f"★ New best model! Loss: {best_loss:.4f}")

        logger.info("=" * 60)
        logger.info("Classifier training completed!")
        logger.info(f"Best validation loss: {best_loss:.4f}")
        logger.info(f"Best validation accuracy: {best_acc:.2f}%")
        logger.info(
            f"Feature weights saved to: {self.output_dir / f'{prefix}_feat_best.pth'}"
        )
        logger.info("=" * 60)


# ==================== PointNetLK Training ====================


class PointNetLKTrainer:
    """Trainer for PointNetLK (Stage 2)."""

    def __init__(self, config, output_dir, transfer_from=None):
        self.config = config
        self.output_dir = output_dir
        self.transfer_from = transfer_from
        self.device = torch.device(config["training"]["device"])

        # PointNet parameters
        self.pointnet_mode = config["training"]["pointnetlk"]["pointnet_mode"]
        self.dim_k = config["model"]["dim_k"]
        symfn_name = config["model"]["symfn"]
        if symfn_name == "max":
            self.sym_fn = ptlk.pointnet.symfn_max
        elif symfn_name == "avg":
            self.sym_fn = ptlk.pointnet.symfn_avg

        # LK parameters
        self.delta = config["model"]["delta"]
        self.learn_delta = config["model"]["learn_delta"]
        self.max_iter = config["model"]["max_iter"]
        self.xtol = config["model"]["xtol"]
        self.p0_zero_mean = config["model"]["p0_zero_mean"]
        self.p1_zero_mean = config["model"]["p1_zero_mean"]
        pointnetlk_cfg = config["training"]["pointnetlk"]
        self.max_train_steps = pointnetlk_cfg.get("max_train_steps")
        self.max_test_steps = pointnetlk_cfg.get("max_test_steps")
        self.grad_clip_norm = pointnetlk_cfg.get("grad_clip_norm")
        self.max_loss = pointnetlk_cfg.get("max_loss")
        self.max_skipped_batches = int(pointnetlk_cfg.get("max_skipped_batches", 100))

    def create_model(self):
        """Create PointNetLK model."""
        # Create PointNet features
        ptnet = ptlk.pointnet.PointNet_features(
            self.dim_k, use_tnet=False, sym_fn=self.sym_fn
        )

        # Load pre-trained weights if specified
        if self.transfer_from and os.path.isfile(self.transfer_from):
            logger.info(f"Loading pre-trained features from: {self.transfer_from}")
            ptnet.load_state_dict(torch.load(self.transfer_from, map_location="cpu"))
        else:
            logger.warning("No transfer learning weights provided!")

        # Freeze or tune PointNet
        if self.pointnet_mode == "fixed":
            logger.info("Freezing PointNet weights")
            for param in ptnet.parameters():
                param.requires_grad_(False)
        elif self.pointnet_mode == "tune":
            logger.info("Fine-tuning PointNet weights")
        else:
            raise ValueError(f"Unknown pointnet_mode: {self.pointnet_mode}")

        # Create PointNetLK
        model = ptlk.pointlk.PointLK(ptnet, self.delta, self.learn_delta)

        logger.info("Created PointNetLK model:")
        logger.info(f"  Max iterations: {self.max_iter}")
        logger.info(f"  Delta: {self.delta}")
        logger.info(f"  Learn delta: {self.learn_delta}")
        logger.info(f"  Zero-mean template: {self.p0_zero_mean}")
        logger.info(f"  Zero-mean source: {self.p1_zero_mean}")

        return model

    def compute_loss(self, model, data, device):
        """Compute registration loss."""
        p0, p1, igt_gt = data
        p0 = p0.to(device)  # template
        p1 = p1.to(device)  # source
        igt_gt = igt_gt.to(device)  # ground truth transform

        # Run PointNetLK using static method do_forward
        r = ptlk.pointlk.PointLK.do_forward(
            model,
            p0,
            p1,
            self.max_iter,
            self.xtol,
            self.p0_zero_mean,
            self.p1_zero_mean,
        )

        # Get estimated transformation
        est_g = model.g

        # Compute loss (combination of residual loss and transformation loss)
        loss_r = ptlk.pointlk.PointLK.rsq(r)  # Residual loss
        loss_g = ptlk.pointlk.PointLK.comp(est_g, igt_gt)  # Transformation loss
        loss = loss_r + loss_g  # Combined loss

        return loss, {
            "loss_r": loss_r.item(),
            "loss_g": loss_g.item(),
            "iters": model.itr,
        }

    def train_epoch(self, model, dataloader, optimizer, device):
        """Train one epoch."""
        model.train()
        total_loss = 0.0
        count = 0
        skipped_batches = 0

        for batch_idx, data in enumerate(dataloader):
            if self.max_train_steps is not None and batch_idx >= self.max_train_steps:
                break
            loss, r = self.compute_loss(model, data, device)
            batch_size = _batch_size_from_points(data[0])
            if not _loss_is_usable(loss, self.max_loss) or r["iters"] < 0:
                skipped_batches += 1
                logger.warning(
                    "Skipping unstable PointNetLK batch "
                    f"{batch_idx}: loss={loss.detach().item()}, "
                    f"loss_r={r['loss_r']}, loss_g={r['loss_g']}, "
                    f"iters={r['iters']}"
                )
                if skipped_batches > self.max_skipped_batches:
                    raise RuntimeError(
                        "PointNetLK training exceeded "
                        f"max_skipped_batches={self.max_skipped_batches}."
                    )
                continue

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.grad_clip_norm,
                )
            optimizer.step()

            # Statistics
            total_loss += loss.item() * batch_size
            count += batch_size

            if batch_idx % self.config["logging"]["log_interval"] == 0:
                logger.info(
                    f"  Batch {batch_idx}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}, "
                    f"Loss_r: {r['loss_r']:.6f}, "
                    f"Loss_g: {r['loss_g']:.6f}, "
                    f"Iters: {r['iters']}"
                )

        if count == 0:
            raise RuntimeError("PointNetLK epoch had no usable training batches.")
        if skipped_batches:
            logger.info(f"Skipped unstable PointNetLK train batches: {skipped_batches}")
        avg_loss = total_loss / count
        return avg_loss

    def eval_epoch(self, model, dataloader, device):
        """Evaluate one epoch."""
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for data in dataloader:
                if (
                    self.max_test_steps is not None
                    and count > 0
                    and count >= self.max_test_steps * _batch_size_from_points(data[0])
                ):
                    break
                loss, r = self.compute_loss(model, data, device)
                batch_size = _batch_size_from_points(data[0])
                if not _loss_is_usable(loss, self.max_loss) or r["iters"] < 0:
                    logger.warning(
                        "Skipping unstable PointNetLK eval batch: "
                        f"loss={loss.detach().item()}, "
                        f"loss_r={r['loss_r']}, loss_g={r['loss_g']}, "
                        f"iters={r['iters']}"
                    )
                    continue
                total_loss += loss.item() * batch_size
                count += batch_size

        if count == 0:
            return float("inf")
        avg_loss = total_loss / count
        return avg_loss

    def train(self, trainset, testset, epochs, dev_mode=False):
        """Main training loop for PointNetLK."""
        logger.info("=" * 60)
        logger.info("Starting PointNetLK Training (Stage 2)")
        logger.info("=" * 60)

        if dev_mode:
            epochs = min(epochs, 3)
            logger.info(f"DEV MODE: Reducing epochs to {epochs}")

        # Create model
        model = self.create_model()
        model.to(self.device)

        # Create tracking datasets with random transforms
        mag = self.config["training"]["pointnetlk"]["mag"]
        from ptlk.data.transforms import RandomTransformSE3

        rigid_transform = RandomTransformSE3(mag=mag, mag_randomly=True)
        shared_perturbation = {
            "perturbation_enabled": self.config["dataset"].get(
                "perturbation_enabled",
                False,
            ),
            "rotation_deg": self.config["dataset"].get("rotation_deg", 0.0),
            "translation_m": self.config["dataset"].get("translation_m", 0.0),
            "noise_sigma": self.config["dataset"].get("noise_sigma", 0.0),
            "noise_clip": self.config["dataset"].get("noise_clip", 0.0),
            "apply_noise_to": self.config["dataset"].get("apply_noise_to", "source"),
        }

        train_tracking = C3VDset4tracking(
            trainset,
            rigid_transform,
            **shared_perturbation,
        )
        test_tracking = C3VDset4tracking(
            testset,
            rigid_transform,
            **shared_perturbation,
        )

        if shared_perturbation["perturbation_enabled"]:
            logger.info(
                "Training with unified perturbation:"
                f" rotation_deg={shared_perturbation['rotation_deg']},"
                f" translation_m={shared_perturbation['translation_m']},"
                f" noise_sigma={shared_perturbation['noise_sigma']},"
                f" noise_clip={shared_perturbation['noise_clip']},"
                f" apply_noise_to={shared_perturbation['apply_noise_to']}"
            )
        else:
            logger.info(f"Training with perturbation magnitude: {mag}")

        # Create dataloaders
        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"]["num_workers"]

        trainloader = torch.utils.data.DataLoader(
            train_tracking,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        testloader = torch.utils.data.DataLoader(
            test_tracking,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Setup optimizer
        pointnetlk_cfg = self.config["training"]["pointnetlk"]
        learning_rate = float(pointnetlk_cfg.get("learning_rate", 1.0e-3))
        learnable_params = [p for p in model.parameters() if p.requires_grad]
        if not learnable_params:
            raise RuntimeError("PointNetLK has no learnable parameters.")
        optimizer = torch.optim.Adam(learnable_params, lr=learning_rate)
        logger.info(f"PointNetLK optimizer: Adam(lr={learning_rate:.2e})")
        if self.grad_clip_norm is not None:
            logger.info(f"PointNetLK gradient clipping: {self.grad_clip_norm}")
        if self.max_loss is not None:
            logger.info(f"PointNetLK max usable loss: {self.max_loss}")

        # Add learning rate scheduler
        scheduler_factor = float(pointnetlk_cfg.get("scheduler_factor", 0.5))
        scheduler_patience = int(pointnetlk_cfg.get("scheduler_patience", 5))
        scheduler_threshold = float(pointnetlk_cfg.get("scheduler_threshold", 1.0e-4))
        scheduler_threshold_mode = str(
            pointnetlk_cfg.get("scheduler_threshold_mode", "rel")
        )
        scheduler_cooldown = int(pointnetlk_cfg.get("scheduler_cooldown", 0))
        scheduler_min_lr = float(pointnetlk_cfg.get("min_lr", 1.0e-8))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            threshold_mode=scheduler_threshold_mode,
            cooldown=scheduler_cooldown,
            min_lr=scheduler_min_lr,
        )
        logger.info(
            "✓ Using ReduceLROnPlateau scheduler "
            f"(factor={scheduler_factor}, patience={scheduler_patience}, "
            f"threshold={scheduler_threshold}, "
            f"threshold_mode={scheduler_threshold_mode}, "
            f"cooldown={scheduler_cooldown}, min_lr={scheduler_min_lr:.2e})"
        )

        # Resume from a PointNetLK training snapshot if requested.
        start_epoch = 0
        best_loss = float("inf")
        resume_from = pointnetlk_cfg.get("resume_from")
        if resume_from:
            resume_path = Path(resume_from).expanduser().resolve()
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"PointNetLK resume checkpoint not found: {resume_path}"
                )
            logger.info(f"Resuming PointNetLK from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
                start_epoch = int(
                    pointnetlk_cfg.get("resume_start_epoch", checkpoint.get("epoch", 0))
                )
                best_loss = float(
                    pointnetlk_cfg.get(
                        "resume_best_test_loss",
                        checkpoint.get("min_loss", best_loss),
                    )
                )
                resume_optimizer = bool(pointnetlk_cfg.get("resume_optimizer", True))
                resume_scheduler = bool(pointnetlk_cfg.get("resume_scheduler", True))
                if "optimizer" in checkpoint and resume_optimizer:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    _move_optimizer_state_to_device(optimizer, self.device)
                    logger.info("✓ Resumed PointNetLK optimizer state")
                elif "optimizer" in checkpoint:
                    logger.info("Resetting PointNetLK optimizer state from config")
                if "scheduler" in checkpoint and resume_scheduler:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    logger.info("✓ Resumed PointNetLK scheduler state")
                elif "scheduler" in checkpoint:
                    logger.info("Resetting PointNetLK scheduler state from config")
            else:
                model.load_state_dict(checkpoint)
                start_epoch = int(pointnetlk_cfg.get("resume_start_epoch", 0))
                best_loss = float(
                    pointnetlk_cfg.get("resume_best_test_loss", best_loss)
                )
            logger.info(
                f"✓ Resume state loaded: start_epoch={start_epoch}, "
                f"best_loss={best_loss:.6f}"
            )

        for epoch in range(start_epoch, epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss = self.train_epoch(model, trainloader, optimizer, self.device)
            logger.info(f"Train Loss: {train_loss:.6f}")

            # Evaluate
            val_loss = self.eval_epoch(model, testloader, self.device)
            logger.info(f"Val Loss: {val_loss:.6f}")

            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(f"Learning Rate: {current_lr:.2e}")

            # Save checkpoints
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            # Save model
            prefix = self.config["output"]["pointnetlk_prefix"]
            checkpoint = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "min_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }

            # Always save last
            save_checkpoint(checkpoint, self.output_dir / f"{prefix}_snap_last.pth")
            save_checkpoint(
                model.state_dict(), self.output_dir / f"{prefix}_model_last.pth"
            )

            if is_best:
                save_checkpoint(checkpoint, self.output_dir / f"{prefix}_snap_best.pth")
                save_checkpoint(
                    model.state_dict(), self.output_dir / f"{prefix}_model_best.pth"
                )
                logger.info(f"★ New best model! Loss: {best_loss:.6f}")

        logger.info("=" * 60)
        logger.info("PointNetLK training completed!")
        logger.info(f"Best validation loss: {best_loss:.6f}")
        logger.info(f"Model saved to: {self.output_dir / f'{prefix}_model_best.pth'}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train PointNetLK on C3VD")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["classifier", "pointnetlk"],
        help="Training stage: classifier or pointnetlk",
    )
    parser.add_argument(
        "--data-root", type=str, required=True, help="Path to C3VD dataset"
    )
    parser.add_argument(
        "--transfer-from",
        type=str,
        default=None,
        help="Path to classifier feature weights (for pointnetlk stage)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to PointNetLK training snapshot for resuming stage 2",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Development mode (fast testing with few epochs)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override data root
    config["dataset"]["data_root"] = args.data_root
    if args.resume_from:
        config["training"]["pointnetlk"]["resume_from"] = args.resume_from

    # Setup output directory
    output_dir, log_dir = setup_output_dir(config, args.stage)

    # Save config
    config_save_path = output_dir / f"config_{args.stage}.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved to: {config_save_path}")

    # Get datasets
    trainset, testset = get_datasets(config, args.data_root)

    # Train
    if args.stage == "classifier":
        trainer = ClassifierTrainer(config, output_dir)
        epochs = config["training"]["classifier"]["epochs"]
        trainer.train(trainset, testset, epochs, dev_mode=args.dev)

    elif args.stage == "pointnetlk":
        if args.transfer_from is None:
            logger.warning("No --transfer-from specified! Training from scratch.")
        trainer = PointNetLKTrainer(
            config, output_dir, transfer_from=args.transfer_from
        )
        epochs = config["training"]["pointnetlk"]["epochs"]
        trainer.train(trainset, testset, epochs, dev_mode=args.dev)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
