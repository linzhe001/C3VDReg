#!/usr/bin/env python
# ruff: noqa: E402,I001
"""
Training script for PointNetLK_Revisited on C3VD dataset.

This script adapts the PointNetLK_Revisited training code to work with C3VD dataset.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.utils.data
import yaml

# Add paths
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parents[1]
REPO_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

# Import PointNetLK_Revisited modules
pointnetlk_revisited_path = REPO_ROOT / "baselines" / "PointNetLK_Revisited"
sys.path.insert(0, str(pointnetlk_revisited_path))

import trainer as plk_trainer

# Import C3VD dataset adapter
from common.datasets.c3vd_for_pointnetlk_revisited import C3VDForPointNetLKRevisited


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class Args:
    """Arguments container compatible with PointNetLK_Revisited."""

    def __init__(self, config):
        # IO settings
        self.outfile = os.path.join(
            config["training"]["checkpoint_dir"], "pointnetlk_c3vd"
        )
        self.dataset_path = config["dataset"]["data_root"]

        # Dataset settings
        self.dataset_type = "c3vd"
        self.data_type = "synthetic"
        self.categoryfile = None
        self.num_points = config["dataset"]["num_points"]
        self.num_random_points = config["pointnetlk"]["num_random_points"]
        self.mag = config["dataset"]["mag"]
        self.sigma = config["dataset"]["sigma"]
        self.clip = config["dataset"]["clip"]
        self.sampling_mode = config["dataset"].get("sampling_mode", "voxel")
        self.normalize_mode = config["dataset"].get("normalize_mode", "none")
        self.perturbation_enabled = config["dataset"].get(
            "perturbation_enabled",
            False,
        )
        self.rotation_deg = config["dataset"].get("rotation_deg", 0.0)
        self.translation_m = config["dataset"].get("translation_m", 0.0)
        self.noise_sigma = config["dataset"].get("noise_sigma", 0.0)
        self.noise_clip = config["dataset"].get("noise_clip", 0.0)
        self.apply_noise_to = config["dataset"].get("apply_noise_to", "source")
        self.workers = config["training"]["num_workers"]

        # Model settings
        self.embedding = config["model"]["embedding"]
        self.dim_k = config["model"]["dim_k"]

        # PointNetLK settings
        self.max_iter = config["pointnetlk"]["max_iter"]

        # Training settings
        self.batch_size = config["training"]["batch_size"]
        self.max_epochs = config["training"]["max_epochs"]
        self.start_epoch = config["training"]["start_epoch"]
        self.optimizer = config["training"]["optimizer"]
        self.device = config["training"]["device"]
        self.lr = config["training"]["lr"]
        self.decay_rate = config["training"]["decay_rate"]
        self.grad_clip_norm = config["training"].get("grad_clip_norm")
        self.max_loss = config["training"].get("max_loss")
        self.eval_max_loss = config["training"].get("eval_max_loss")
        self.max_skipped_batches = int(
            config["training"].get("max_skipped_batches", 100)
        )

        # Logging
        self.logfile = os.path.join(
            config["training"]["checkpoint_dir"], "training.log"
        )

        # Resume
        self.resume = config["resume"].get("checkpoint", "")
        self.pretrained = config["resume"].get("pretrained", "")

        # Additional config
        self.config = config


def _init_(config):
    """Initialize directories and backup files."""
    checkpoint_dir = config["training"]["checkpoint_dir"]

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Backup this script
    backup_file = os.path.join(
        checkpoint_dir, "train_pointnetlk_revisited_c3vd.py.backup"
    )
    os.system(f"cp {__file__} {backup_file}")

    # Save config
    config_file = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Config saved to: {config_file}")


def _loss_is_usable(loss, max_loss=None):
    if not torch.isfinite(loss):
        return False
    if max_loss is not None and float(loss.detach().item()) > float(max_loss):
        return False
    return True


def _train_one_epoch_guarded(
    dptnetlk,
    ptnetlk,
    trainloader,
    optimizer,
    args,
):
    ptnetlk.float()
    ptnetlk.train()
    total_loss = 0.0
    total_pose_loss = 0.0
    usable_batches = 0
    skipped_batches = 0

    for batch_idx, data in enumerate(trainloader):
        loss, loss_pose = dptnetlk.compute_loss(
            ptnetlk,
            data,
            args.device,
            "train",
            args.data_type,
            args.num_random_points,
        )
        if not _loss_is_usable(loss, args.max_loss):
            skipped_batches += 1
            LOGGER.warning(
                "Skipping unstable PointNetLK_Revisited train batch %d: loss=%s",
                batch_idx,
                loss.detach().item(),
            )
            if skipped_batches > args.max_skipped_batches:
                raise RuntimeError(
                    "PointNetLK_Revisited training exceeded "
                    f"max_skipped_batches={args.max_skipped_batches}."
                )
            continue

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                ptnetlk.parameters(),
                float(args.grad_clip_norm),
            )
        optimizer.step()

        total_loss += float(loss.detach().item())
        total_pose_loss += float(loss_pose.detach().item())
        usable_batches += 1

    if usable_batches == 0:
        raise RuntimeError("PointNetLK_Revisited epoch had no usable training batches.")
    if skipped_batches:
        LOGGER.info(
            "Skipped unstable PointNetLK_Revisited train batches: %d",
            skipped_batches,
        )
    return total_loss / usable_batches, total_pose_loss / usable_batches


def _eval_one_epoch_guarded(dptnetlk, ptnetlk, evalloader, args):
    ptnetlk.eval()
    total_loss = 0.0
    total_pose_loss = 0.0
    usable_batches = 0
    skipped_batches = 0

    for batch_idx, data in enumerate(evalloader):
        loss, loss_pose = dptnetlk.compute_loss(
            ptnetlk,
            data,
            args.device,
            "eval",
            args.data_type,
            args.num_random_points,
        )
        if not _loss_is_usable(loss, args.eval_max_loss):
            skipped_batches += 1
            LOGGER.warning(
                "Skipping unstable PointNetLK_Revisited eval batch %d: loss=%s",
                batch_idx,
                loss.detach().item(),
            )
            continue

        total_loss += float(loss.detach().item())
        total_pose_loss += float(loss_pose.detach().item())
        usable_batches += 1

    if skipped_batches:
        LOGGER.info(
            "Skipped unstable PointNetLK_Revisited eval batches: %d",
            skipped_batches,
        )
    if usable_batches == 0:
        return float("inf"), float("inf")
    return total_loss / usable_batches, total_pose_loss / usable_batches


def get_datasets(args):
    """Create train and test datasets for C3VD."""
    train_ratio = args.config["dataset"]["train_ratio"]
    random_seed = args.config["dataset"]["random_seed"]
    dataset_kwargs = {
        "train_scenes": args.config["dataset"].get("train_scenes"),
        "val_scenes": args.config["dataset"].get("val_scenes"),
        "test_scenes": args.config["dataset"].get("test_scenes"),
        "frame_stride": args.config["dataset"].get("frame_stride", 1),
    }

    print("\n" + "=" * 60)
    print("Creating C3VD datasets...")
    print("=" * 60)

    # Create training dataset
    trainset = C3VDForPointNetLKRevisited(
        data_root=args.dataset_path,
        num_points=args.num_points,
        split="train",
        mag=args.mag,
        sigma=args.sigma,
        clip=args.clip,
        sampling_mode=args.sampling_mode,
        normalize_mode=args.normalize_mode,
        perturbation_enabled=args.perturbation_enabled,
        rotation_deg=args.rotation_deg,
        translation_m=args.translation_m,
        noise_sigma=args.noise_sigma,
        noise_clip=args.noise_clip,
        apply_noise_to=args.apply_noise_to,
        train_ratio=train_ratio,
        random_seed=random_seed,
        max_pairs=args.config["dataset"].get("max_train_pairs"),
        **dataset_kwargs,
    )

    # Create evaluation dataset
    evalset = C3VDForPointNetLKRevisited(
        data_root=args.dataset_path,
        num_points=args.num_points,
        split="val",
        mag=args.mag,
        sigma=args.sigma,
        clip=args.clip,
        sampling_mode=args.sampling_mode,
        normalize_mode=args.normalize_mode,
        perturbation_enabled=args.perturbation_enabled,
        rotation_deg=args.rotation_deg,
        translation_m=args.translation_m,
        noise_sigma=args.noise_sigma,
        noise_clip=args.noise_clip,
        apply_noise_to=args.apply_noise_to,
        train_ratio=train_ratio,
        random_seed=random_seed,
        max_pairs=args.config["dataset"].get("max_val_pairs")
        or args.config["dataset"].get("max_test_pairs"),
        **dataset_kwargs,
    )

    print("\n" + "=" * 60)
    print(f"Training set size: {len(trainset)}")
    print(f"Evaluation set size: {len(evalset)}")
    print("=" * 60 + "\n")

    return trainset, evalset


def train(args, trainset, evalset, dptnetlk):
    """Main training loop."""
    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device = torch.device(args.device)

    print(f"\nUsing device: {args.device}")
    max_train_steps = args.config["training"].get("max_train_steps")
    max_val_steps = args.config["training"].get("max_val_steps")

    # Create model
    model = dptnetlk.create_model()

    # Load pretrained weights if specified
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        print(f"Loading pretrained weights from: {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location="cpu"))

    model.to(args.device)

    # Resume from checkpoint if specified
    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
    print(f"Starting from epoch {args.start_epoch + 1}")

    # Create data loaders
    evalloader = torch.utils.data.DataLoader(
        evalset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )

    print("\nData loaders created:")
    print(f"  Training batches: {len(trainloader)}")
    print(f"  Evaluation batches: {len(evalloader)}")

    # Initialize tracking variables
    min_loss = float("inf")
    min_info = float("inf")

    # Setup optimizer
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            learnable_params, lr=args.lr, weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=args.lr)

    # Resume optimizer state if available
    if checkpoint is not None:
        min_loss = checkpoint["min_loss"]
        min_info = checkpoint["min_info"]
        optimizer.load_state_dict(checkpoint["optimizer"])

    # Add learning rate scheduler to prevent loss explosion
    # ReduceLROnPlateau: reduce lr when validation loss plateaus or increases
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # minimize validation loss
        factor=float(args.config["training"].get("scheduler_factor", 0.5)),
        patience=int(args.config["training"].get("scheduler_patience", 5)),
        threshold=float(args.config["training"].get("scheduler_threshold", 1.0e-4)),
        threshold_mode=str(
            args.config["training"].get("scheduler_threshold_mode", "rel")
        ),
        cooldown=int(args.config["training"].get("scheduler_cooldown", 0)),
        min_lr=float(args.config["training"].get("min_lr", 1.0e-8)),
    )
    print("✓ Using ReduceLROnPlateau scheduler")
    print(
        "  - Learning rate will be reduced by 50% if validation loss "
        "doesn't improve for 5 epochs"
    )
    print(f"  - Minimum learning rate: {args.config['training'].get('min_lr', 1.0e-8)}")

    # Resume scheduler state if available
    if checkpoint is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    print("✓ Resumed scheduler state from checkpoint")
    if args.max_loss is not None:
        print(f"✓ Using train max usable loss guard: {args.max_loss}")
    if args.eval_max_loss is not None:
        print(f"✓ Using eval max usable loss guard: {args.eval_max_loss}")
    if args.grad_clip_norm is not None:
        print(f"✓ Using gradient clipping: {args.grad_clip_norm}")

    # Training loop
    print("\n" + "=" * 60)
    print("Begin Training!")
    print("=" * 60 + "\n")

    for epoch in range(args.start_epoch, args.max_epochs):
        print(f"Epoch [{epoch + 1}/{args.max_epochs}]")

        # Training
        if max_train_steps is not None:
            limited_trainloader = list(iter(trainloader))[: int(max_train_steps)]
        else:
            limited_trainloader = trainloader

        if args.max_loss is None and args.grad_clip_norm is None:
            running_loss, running_info = dptnetlk.train_one_epoch(
                model,
                limited_trainloader,
                optimizer,
                args.device,
                "train",
                args.data_type,
                num_random_points=args.num_random_points,
            )
        else:
            running_loss, running_info = _train_one_epoch_guarded(
                dptnetlk,
                model,
                limited_trainloader,
                optimizer,
                args,
            )

        # Evaluation
        if max_val_steps is not None:
            limited_evalloader = list(iter(evalloader))[: int(max_val_steps)]
        else:
            limited_evalloader = evalloader

        if args.eval_max_loss is None:
            val_loss, val_info = dptnetlk.eval_one_epoch(
                model,
                limited_evalloader,
                args.device,
                "eval",
                args.data_type,
                num_random_points=args.num_random_points,
            )
        else:
            val_loss, val_info = _eval_one_epoch_guarded(
                dptnetlk,
                model,
                limited_evalloader,
                args,
            )

        # Check if best model
        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Log results
        LOGGER.info(
            "epoch, %04d, %f, %f, %f, %f",
            epoch + 1,
            running_loss,
            val_loss,
            running_info,
            val_info,
        )

        print(f"  Train Loss: {running_loss:.6f}, Train Info: {running_info:.6f}")
        print(f"  Val Loss: {val_loss:.6f}, Val Info: {val_info:.6f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        if is_best:
            print(f"  *** New best model! (loss: {min_loss:.6f}) ***")
        print()

        # Save checkpoint
        snap = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "min_loss": min_loss,
            "min_info": min_info,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

        # Save best model
        if is_best:
            best_model_path = f"{args.outfile}_model_best.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model to: {best_model_path}")

        # Save latest checkpoint
        latest_snap_path = f"{args.outfile}_snap_last.pth"
        torch.save(snap, latest_snap_path)

        # Save checkpoint at intervals
        save_interval = args.config["training"].get("save_interval", 10)
        if (epoch + 1) % save_interval == 0:
            epoch_snap_path = f"{args.outfile}_snap_epoch_{epoch + 1:04d}.pth"
            torch.save(snap, epoch_snap_path)
            print(f"  Saved checkpoint to: {epoch_snap_path}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {min_loss:.6f}")
    print("=" * 60 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train PointNetLK_Revisited on C3VD dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/benchmarking/bridges/configs/c3vd_pointnetlk_revisited.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-root", type=str, default=None, help="Override data root directory"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Override number of workers"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args_parsed = parser.parse_args()

    # Load config
    print(f"Loading config from: {args_parsed.config}")
    with open(args_parsed.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    if args_parsed.data_root is not None:
        config["dataset"]["data_root"] = args_parsed.data_root
    if args_parsed.device is not None:
        config["training"]["device"] = args_parsed.device
    if args_parsed.batch_size is not None:
        config["training"]["batch_size"] = args_parsed.batch_size
    if args_parsed.num_workers is not None:
        config["training"]["num_workers"] = args_parsed.num_workers
    if args_parsed.epochs is not None:
        config["training"]["max_epochs"] = args_parsed.epochs
    if args_parsed.resume is not None:
        config["resume"]["checkpoint"] = args_parsed.resume

    # Create args object
    args = Args(config)

    # Initialize
    _init_(config)

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s:%(name)s, %(asctime)s, %(message)s",
        filename=args.logfile,
    )
    LOGGER.debug("Training (PID=%d), %s", os.getpid(), args.config)

    # Create datasets
    trainset, evalset = get_datasets(args)

    # Create trainer
    dptnetlk = plk_trainer.TrainerAnalyticalPointNetLK(args)

    # Train
    train(args, trainset, evalset, dptnetlk)

    LOGGER.debug("Training completed! (PID=%d)", os.getpid())
    print("All done!")


if __name__ == "__main__":
    main()
