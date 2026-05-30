#!/usr/bin/env python
# ruff: noqa: E402,I001
"""
Training script for RegTR on C3VD dataset.

This script adapts the RegTR training code to work with C3VD dataset.
Unlike the original RegTR which uses pre-computed h5 files, this version
computes overlap on-the-fly for the C3VD dataset.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from easydict import EasyDict

# Add paths
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parents[1]
REPO_ROOT = SRC_ROOT.parent
regtr_src_dir = REPO_ROOT / "baselines" / "RegTR" / "src"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(regtr_src_dir))


def _module_origin(module):
    module_file = getattr(module, "__file__", None)
    if module_file:
        return os.path.abspath(module_file)

    module_path = getattr(module, "__path__", None)
    if module_path:
        try:
            return os.path.abspath(next(iter(module_path)))
        except StopIteration:
            return ""
    return ""


def _purge_conflicting_regtr_modules(expected_root):
    managed_prefixes = (
        "benchmark",
        "cvhelpers",
        "data_loaders",
        "models",
        "trainer",
        "utils",
    )
    expected_root = os.path.abspath(expected_root)
    for module_name, module in list(sys.modules.items()):
        if module_name.split(".", 1)[0] not in managed_prefixes:
            continue
        origin = _module_origin(module)
        if origin and not origin.startswith(expected_root):
            sys.modules.pop(module_name, None)


_purge_conflicting_regtr_modules(regtr_src_dir)

# Import RegTR modules
try:
    from cvhelpers.misc import prepare_logger
    from cvhelpers.torch_helpers import setup_seed
    from models import get_model
    from trainer import Trainer
except ImportError as e:
    print(f"Error importing RegTR modules: {e}")
    print("Make sure RegTR dependencies are installed:")
    print("  cd RegTR/src")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Import C3VD dataset adapter
from common.datasets.c3vd_for_regtr import C3VDForRegTR

import torch
import torchvision


class C3VDDataLoaderWrapper:
    """Wrapper to integrate C3VD dataset with RegTR's data loading pipeline."""

    def __init__(self, cfg, phase="train", num_workers=0):
        """
        Args:
            cfg: EasyDict config object
            phase: 'train', 'val', or 'test'
            num_workers: Number of data loading workers
        """
        self.cfg = cfg
        self.phase = phase
        self.num_workers = num_workers

        # Determine split based on phase
        if phase not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported phase: {phase}")
        split = phase
        if phase == "train":
            max_pairs = getattr(cfg, "max_train_pairs", None)
        elif phase == "val":
            max_pairs = getattr(cfg, "max_val_pairs", None) or getattr(
                cfg,
                "max_test_pairs",
                None,
            )
        else:
            max_pairs = getattr(cfg, "max_test_pairs", None)

        # Create dataset
        self.dataset = C3VDForRegTR(
            data_root=cfg.data_root,
            split=split,
            num_points=cfg.num_points,
            overlap_radius=cfg.overlap_radius,
            train_ratio=cfg.train_ratio,
            random_seed=cfg.random_seed,
            sampling_mode=getattr(cfg, "sampling_mode", "voxel"),
            normalize_mode=getattr(cfg, "normalize_mode", "none"),
            perturbation_enabled=getattr(cfg, "perturbation_enabled", False),
            rotation_deg=getattr(cfg, "rotation_deg", 0.0),
            translation_m=getattr(cfg, "translation_m", 0.0),
            noise_sigma=getattr(cfg, "noise_sigma", 0.0),
            noise_clip=getattr(cfg, "noise_clip", 0.0),
            apply_noise_to=getattr(cfg, "apply_noise_to", "source"),
            train_scenes=getattr(cfg, "train_scenes", None),
            val_scenes=getattr(cfg, "val_scenes", None),
            test_scenes=getattr(cfg, "test_scenes", None),
            frame_stride=getattr(cfg, "frame_stride", 1),
            max_pairs=max_pairs,
        )

        # Apply data augmentation for training
        if phase == "train" and hasattr(cfg, "augment_noise"):
            # Import transforms from RegTR
            import data_loaders.transforms as regtr_transforms

            transforms_list = []
            if getattr(cfg, "perturbation_enabled", False):
                perturb_pose = None
                augment_noise = 0.0
            else:
                perturb_pose = getattr(cfg, "perturb_pose", None)
                augment_noise = cfg.augment_noise
            if perturb_pose:
                transforms_list.append(
                    regtr_transforms.RigidPerturb(perturb_mode=perturb_pose)
                )
            if augment_noise > 0:
                transforms_list.append(
                    regtr_transforms.Jitter(scale=augment_noise)
                )
            transforms_list.extend(
                [
                    regtr_transforms.ShufflePoints(),
                    regtr_transforms.RandomSwap(),
                ]
            )

            self.transforms = torchvision.transforms.Compose(transforms_list)
        else:
            self.transforms = None

    def get_dataloader(self):
        """Create and return the DataLoader."""
        from data_loaders.collate_functions import collate_pair

        batch_size = getattr(self.cfg, f"{self.phase}_batch_size")
        shuffle = self.phase == "train"

        # Wrap dataset with transform if needed
        if self.transforms is not None:
            # Create a simple wrapper that applies transforms
            class TransformedDataset(torch.utils.data.Dataset):
                def __init__(self, base_dataset, transforms):
                    self.base_dataset = base_dataset
                    self.transforms = transforms

                def __len__(self):
                    return len(self.base_dataset)

                def __getitem__(self, idx):
                    item = self.base_dataset[idx]
                    if self.transforms is not None:
                        item = self.transforms(item)
                    return item

            dataset = TransformedDataset(self.dataset, self.transforms)
        else:
            dataset = self.dataset

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_pair,
            pin_memory=True,
        )

        return data_loader


def get_c3vd_dataloader(cfg, phase="train", num_workers=0):
    """
    DataLoader factory function compatible with RegTR's get_dataloader interface.

    Args:
        cfg: EasyDict config object
        phase: 'train', 'val', or 'test'
        num_workers: Number of data loading workers

    Returns:
        torch.utils.data.DataLoader
    """
    wrapper = C3VDDataLoaderWrapper(cfg, phase, num_workers)
    return wrapper.get_dataloader()


def _flatten_regtr_config(config_dict):
    """Flatten nested RegTR config sections while preserving top-level scalars."""
    flattened = {}
    for section, value in config_dict.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[section] = value
    return flattened


def main():
    # Argument parsing (matching RegTR's interface)
    parser = argparse.ArgumentParser(description="Train RegTR on C3VD dataset")

    # General
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
    )

    # Logging
    parser.add_argument(
        "--logdir",
        type=str,
        default="../logs",
        help="Directory to store logs, summaries, checkpoints.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="If true, will ignore logdir and log to ../logdev instead",
    )
    parser.add_argument(
        "--name", type=str, help="Experiment name (used to name output directory)"
    )
    parser.add_argument(
        "--summary_every",
        type=int,
        default=500,
        help="Interval to save tensorboard summaries",
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=-1,
        help="Validation interval. Default: every epoch",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, will enable autograd anomaly detection",
    )

    # Misc
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for dataloader",
    )
    parser.add_argument("--data-root", type=str, help="Override data root path")

    # Training and model options
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument(
        "--nb_sanity_val_steps",
        type=int,
        default=2,
        help="Number of validation sanity steps to run before training.",
    )

    opt = parser.parse_args()

    # Load config
    if not os.path.exists(opt.config):
        print(f"Config file not found: {opt.config}")
        sys.exit(1)

    with open(opt.config, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = EasyDict(_flatten_regtr_config(config_dict))

    # Override data root if provided
    if opt.data_root:
        cfg.data_root = opt.data_root

    # Setup experiment name
    if opt.name is None and "expt_name" in cfg:
        opt.name = cfg.expt_name
    elif opt.name is None:
        opt.name = "c3vd_regtr"

    # Hack: Store C3VD experiments in its own subdirectory
    opt.logdir = os.path.join(opt.logdir, "c3vd")

    # Prepare logger
    logger, opt.log_path = prepare_logger(opt)
    logger.info(f"Logging to: {opt.log_path}")

    # Save config to log directory
    config_out_fname = os.path.join(opt.log_path, "config.yaml")
    with open(config_out_fname, "w") as out_fid:
        out_fid.write("# C3VD RegTR Training Config\n")
        out_fid.write(f"# Original file: {opt.config}\n\n")
        yaml.dump(config_dict, out_fid, default_flow_style=False)

    logger.info("Config saved to: {}".format(config_out_fname))

    # Set random seed
    if "seed" in cfg:
        setup_seed(cfg.seed, cudnn_deterministic=False)
        logger.info(f"Random seed set to: {cfg.seed}")

    # Create data loaders
    logger.info("\n" + "=" * 60)
    logger.info("Creating data loaders...")
    logger.info("=" * 60)

    train_loader = get_c3vd_dataloader(cfg, phase="train", num_workers=opt.num_workers)
    val_loader = get_c3vd_dataloader(cfg, phase="val", num_workers=opt.num_workers)

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    logger.info("\n" + "=" * 60)
    logger.info("Creating model...")
    logger.info("=" * 60)

    Model = get_model(cfg.model)
    model = Model(cfg)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    # Extract training parameters
    niter = cfg.niter if "niter" in cfg else -50
    grad_clip = cfg.grad_clip if "grad_clip" in cfg else 0.0

    trainer = Trainer(opt, niter=niter, grad_clip=grad_clip)

    # Start training
    try:
        trainer.fit(model, train_loader, val_loader)
        logger.info("\n" + "=" * 60)
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
