import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from scipy.spatial.transform import Rotation
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add paths
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parents[1]
REPO_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baselines" / "dcp"))

# Import DCP modules
from model import DCP  # noqa: E402

# Import C3VD dataset adapter
from common.datasets.c3vd_for_dcp import C3VDForDCP  # noqa: E402


def _rotation_matrices_to_euler_deg(
    matrices: np.ndarray,
    seq: str = "zyx",
) -> np.ndarray:
    """Convert rotation matrices to Euler angles with modern SciPy APIs."""

    return Rotation.from_matrix(matrices).as_euler(seq, degrees=True).astype("float32")


def _load_model_state(path: str | os.PathLike[str], device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
    return checkpoint


def _dcp_supervised_pose_loss(
    rotation_ab_pred: torch.Tensor,
    translation_ab_pred: torch.Tensor,
    rotation_ab: torch.Tensor,
    translation_ab: torch.Tensor,
    rotation_ba_pred: torch.Tensor,
    translation_ba_pred: torch.Tensor,
    use_cycle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original DCP pose supervision: MSE on SE(3), not point-order MSE."""

    batch_size = rotation_ab.size(0)
    identity = torch.eye(
        3,
        device=rotation_ab.device,
        dtype=rotation_ab.dtype,
    ).unsqueeze(0).repeat(batch_size, 1, 1)

    pose_loss = F.mse_loss(
        torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab),
        identity,
    ) + F.mse_loss(translation_ab_pred, translation_ab)

    cycle_loss = rotation_ab.new_tensor(0.0)
    if use_cycle:
        rotation_cycle_loss = F.mse_loss(
            torch.matmul(rotation_ba_pred, rotation_ab_pred),
            identity,
        )
        translation_cycle_loss = torch.mean(
            (
                torch.matmul(
                    rotation_ba_pred.transpose(2, 1),
                    translation_ab_pred.view(batch_size, 3, 1),
                ).view(batch_size, 3)
                + translation_ba_pred
            )
            ** 2,
            dim=[0, 1],
        )
        cycle_loss = rotation_cycle_loss + translation_cycle_loss
        pose_loss = pose_loss + cycle_loss * 0.1

    return pose_loss, cycle_loss


def _maybe_resume_training(config, net, device, textio):
    training_config = config["training"]
    resume_from = training_config.get("resume_from")
    if not resume_from:
        return 0, np.inf

    resume_path = Path(str(resume_from)).expanduser()
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    net.load_state_dict(_load_model_state(resume_path, device))
    start_epoch = int(training_config.get("resume_start_epoch", 0))
    if start_epoch < 0:
        raise ValueError("training.resume_start_epoch must be non-negative.")

    raw_best_loss = training_config.get("resume_best_test_loss")
    best_test_loss = np.inf if raw_best_loss is None else float(raw_best_loss)
    textio.cprint(
        "Resumed model weights from "
        f"{resume_path} at completed epoch {start_epoch}"
    )
    if np.isfinite(best_test_loss):
        textio.cprint(f"Resumed best test loss: {best_test_loss:.6f}")
    return start_epoch, best_test_loss


class IOStream:
    """Simple IO stream for logging."""

    def __init__(self, path):
        self.f = open(path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args, config):
    """Initialize directories and backup files."""
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    checkpoint_dir = config["training"]["checkpoint_dir"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.join(checkpoint_dir, "models")):
        os.makedirs(os.path.join(checkpoint_dir, "models"))

    # Backup this script
    os.system(f"cp {__file__} {checkpoint_dir}/train_dcp_c3vd.py.backup")

    # Save config
    with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)


def test_one_epoch(args, net, test_loader, device):
    """Test for one epoch."""
    net.eval()

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    max_test_steps = getattr(args, "max_test_steps", None)

    for step_idx, (
        src,
        target,
        rotation_ab,
        translation_ab,
        rotation_ba,
        translation_ba,
        euler_ab,
        euler_ba,
    ) in enumerate(tqdm(test_loader), start=1):
        src = src.to(device)
        target = target.to(device)
        rotation_ab = rotation_ab.to(device)
        translation_ab = translation_ab.to(device)
        rotation_ba = rotation_ba.to(device)
        translation_ba = translation_ba.to(device)

        batch_size = src.size(0)
        num_examples += batch_size

        with torch.no_grad():
            (
                rotation_ab_pred,
                translation_ab_pred,
                rotation_ba_pred,
                translation_ba_pred,
            ) = net(src, target)

        # Compute losses
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())

        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        loss, cycle_loss = _dcp_supervised_pose_loss(
            rotation_ab_pred=rotation_ab_pred,
            translation_ab_pred=translation_ab_pred,
            rotation_ab=rotation_ab,
            translation_ab=translation_ab,
            rotation_ba_pred=rotation_ba_pred,
            translation_ba_pred=translation_ba_pred,
            use_cycle=args.cycle,
        )

        total_loss += loss.item() * batch_size
        total_cycle_loss += cycle_loss.item() * batch_size

        if max_test_steps is not None and step_idx >= max_test_steps:
            break

    # Concatenate all results
    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
    eulers_ab = np.concatenate(eulers_ab, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    # Compute rotation errors
    eulers_ab_pred = _rotation_matrices_to_euler_deg(rotations_ab_pred)
    eulers_ba_pred = _rotation_matrices_to_euler_deg(rotations_ba_pred)
    eulers_ab_gt = np.degrees(eulers_ab)
    eulers_ba_gt = np.degrees(eulers_ba)

    r_ab_mse = np.mean((eulers_ab_gt - eulers_ab_pred) ** 2)
    r_ab_rmse = np.sqrt(r_ab_mse)
    r_ab_mae = np.mean(np.abs(eulers_ab_gt - eulers_ab_pred))

    t_ab_mse = np.mean((translations_ab - translations_ab_pred) ** 2)
    t_ab_rmse = np.sqrt(t_ab_mse)
    t_ab_mae = np.mean(np.abs(translations_ab - translations_ab_pred))

    r_ba_mse = np.mean((eulers_ba_gt - eulers_ba_pred) ** 2)
    r_ba_rmse = np.sqrt(r_ba_mse)
    r_ba_mae = np.mean(np.abs(eulers_ba_gt - eulers_ba_pred))

    t_ba_mse = np.mean((translations_ba - translations_ba_pred) ** 2)
    t_ba_rmse = np.sqrt(t_ba_mse)
    t_ba_mae = np.mean(np.abs(translations_ba - translations_ba_pred))

    return {
        "loss": total_loss / num_examples,
        "cycle_loss": total_cycle_loss / num_examples,
        "r_ab_mse": r_ab_mse,
        "r_ab_rmse": r_ab_rmse,
        "r_ab_mae": r_ab_mae,
        "t_ab_mse": t_ab_mse,
        "t_ab_rmse": t_ab_rmse,
        "t_ab_mae": t_ab_mae,
        "r_ba_mse": r_ba_mse,
        "r_ba_rmse": r_ba_rmse,
        "r_ba_mae": r_ba_mae,
        "t_ba_mse": t_ba_mse,
        "t_ba_rmse": t_ba_rmse,
        "t_ba_mae": t_ba_mae,
    }


def train_one_epoch(args, net, train_loader, opt, device):
    """Train for one epoch."""
    net.train()

    total_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    max_train_steps = getattr(args, "max_train_steps", None)

    for step_idx, (
        src,
        target,
        rotation_ab,
        translation_ab,
        rotation_ba,
        translation_ba,
        euler_ab,
        euler_ba,
    ) in enumerate(tqdm(train_loader), start=1):
        src = src.to(device)
        target = target.to(device)
        rotation_ab = rotation_ab.to(device)
        translation_ab = translation_ab.to(device)
        rotation_ba = rotation_ba.to(device)
        translation_ba = translation_ba.to(device)

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size

        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = (
            net(src, target)
        )

        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())

        loss, _ = _dcp_supervised_pose_loss(
            rotation_ab_pred=rotation_ab_pred,
            translation_ab_pred=translation_ab_pred,
            rotation_ab=rotation_ab,
            translation_ab=translation_ab,
            rotation_ba_pred=rotation_ba_pred,
            translation_ba_pred=translation_ba_pred,
            use_cycle=args.cycle,
        )

        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_size

        if max_train_steps is not None and step_idx >= max_train_steps:
            break

    avg_loss = total_loss / num_examples

    return {"loss": avg_loss}


def train(args, config):
    """Main training function."""
    # Set random seed
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    # Setup device
    if config["training"]["use_cuda"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['training']['cuda_device']}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = C3VDForDCP(
        data_root=config["dataset"]["data_root"],
        num_points=config["dataset"]["num_points"],
        split="train",
        gaussian_noise=config["dataset"]["gaussian_noise"],
        rot_factor=config["dataset"]["rot_factor"],
        trans_mag=config["dataset"]["trans_mag"],
        sampling_mode=config["dataset"].get("sampling_mode", "voxel"),
        normalize_mode=config["dataset"].get("normalize_mode", "unit_cube"),
        perturbation_enabled=config["dataset"].get("perturbation_enabled", False),
        rotation_deg=config["dataset"].get("rotation_deg", 0.0),
        translation_m=config["dataset"].get("translation_m", 0.0),
        noise_sigma=config["dataset"].get("noise_sigma", 0.0),
        noise_clip=config["dataset"].get("noise_clip", 0.0),
        apply_noise_to=config["dataset"].get("apply_noise_to", "source"),
        train_ratio=config["dataset"]["train_ratio"],
        random_seed=config["dataset"]["random_seed"],
        train_scenes=config["dataset"].get("train_scenes"),
        val_scenes=config["dataset"].get("val_scenes"),
        test_scenes=config["dataset"].get("test_scenes"),
        frame_stride=config["dataset"].get("frame_stride", 1),
        max_pairs=config["dataset"].get("max_train_pairs"),
    )

    val_dataset = C3VDForDCP(
        data_root=config["dataset"]["data_root"],
        num_points=config["dataset"]["num_points"],
        split="val",
        gaussian_noise=False,  # No noise for testing
        rot_factor=config["dataset"]["rot_factor"],
        trans_mag=config["dataset"]["trans_mag"],
        sampling_mode=config["dataset"].get("sampling_mode", "voxel"),
        normalize_mode=config["dataset"].get("normalize_mode", "unit_cube"),
        perturbation_enabled=config["dataset"].get("perturbation_enabled", False),
        rotation_deg=config["dataset"].get("rotation_deg", 0.0),
        translation_m=config["dataset"].get("translation_m", 0.0),
        noise_sigma=config["dataset"].get("noise_sigma", 0.0),
        noise_clip=config["dataset"].get("noise_clip", 0.0),
        apply_noise_to=config["dataset"].get("apply_noise_to", "source"),
        train_ratio=config["dataset"]["train_ratio"],
        random_seed=config["dataset"]["random_seed"],
        train_scenes=config["dataset"].get("train_scenes"),
        val_scenes=config["dataset"].get("val_scenes"),
        test_scenes=config["dataset"].get("test_scenes"),
        frame_stride=config["dataset"].get("frame_stride", 1),
        max_pairs=config["dataset"].get("max_val_pairs")
        or config["dataset"].get("max_test_pairs"),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config["training"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["test_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config["training"]["num_workers"],
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    net = DCP(args).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        net = nn.DataParallel(net)

    # Create optimizer
    if config["training"]["optimizer"] == "SGD":
        opt = optim.SGD(
            net.parameters(),
            lr=config["training"]["lr"],
            momentum=config["training"]["momentum"],
        )
    else:
        opt = optim.Adam(net.parameters(), lr=config["training"]["lr"])

    # Create scheduler
    scheduler = MultiStepLR(
        opt,
        milestones=config["training"]["lr_decay_epochs"],
        gamma=config["training"]["lr_decay_rate"],
    )

    # Create logger
    checkpoint_dir = config["training"]["checkpoint_dir"]
    textio = IOStream(os.path.join(checkpoint_dir, "run.log"))
    textio.cprint(str(args))
    textio.cprint(str(config))

    start_epoch, best_test_loss = _maybe_resume_training(config, net, device, textio)

    # Training loop
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        textio.cprint(
            f"\n==== Epoch {epoch + 1}/{config['training']['num_epochs']} ===="
        )

        # Train
        train_stats = train_one_epoch(args, net, train_loader, opt, device)
        textio.cprint(f"Train - Loss: {train_stats['loss']:.6f}")

        # Test
        if (epoch + 1) % config["training"]["val_interval"] == 0:
            test_stats = test_one_epoch(args, net, val_loader, device)
            textio.cprint(
                f"Val   - Loss: {test_stats['loss']:.6f}, "
                f"R_MAE: {test_stats['r_ab_mae']:.4f}, "
                f"T_MAE: {test_stats['t_ab_mae']:.4f}"
            )

            # Save best model
            if test_stats["loss"] < best_test_loss:
                best_test_loss = test_stats["loss"]
                torch.save(
                    net.state_dict(),
                    os.path.join(checkpoint_dir, "models", "model_best.pth"),
                )
                textio.cprint(f"Saved best model (loss: {best_test_loss:.6f})")

        # Save checkpoint
        if (epoch + 1) % config["training"]["save_interval"] == 0:
            torch.save(
                net.state_dict(),
                os.path.join(checkpoint_dir, "models", f"model_epoch_{epoch + 1}.pth"),
            )
            textio.cprint(f"Saved checkpoint at epoch {epoch + 1}")

        # Update learning rate
        scheduler.step()

    textio.cprint("\n==== Training completed! ====")
    textio.cprint(f"Best test loss: {best_test_loss:.6f}")
    textio.close()


def main():
    parser = argparse.ArgumentParser(description="Train DCP on C3VD dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data-root", type=str, help="Override data root path")

    # DCP model arguments
    parser.add_argument(
        "--emb_nn",
        type=str,
        default="dgcnn",
        metavar="N",
        choices=["pointnet", "dgcnn"],
        help="Embedding network: pointnet or dgcnn",
    )
    parser.add_argument(
        "--pointer",
        type=str,
        default="transformer",
        metavar="N",
        choices=["identity", "transformer"],
        help="Pointer network: identity or transformer",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="svd",
        metavar="N",
        choices=["mlp", "svd"],
        help="Head to use: mlp or svd",
    )
    parser.add_argument(
        "--emb_dims", type=int, default=512, metavar="N", help="Dimension of embeddings"
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=1,
        metavar="N",
        help="Number of blocks of encoder&decoder",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=4,
        metavar="N",
        help="Number of heads in multiheadedattention",
    )
    parser.add_argument(
        "--ff_dims", type=int, default=1024, metavar="N", help="Feed forward dimensions"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, metavar="N", help="Dropout ratio"
    )
    parser.add_argument(
        "--cycle",
        type=bool,
        default=False,
        metavar="N",
        help="Whether to use cycle consistency",
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override with command line arguments
    if args.data_root:
        config["dataset"]["data_root"] = args.data_root

    # Update args with config
    for key, value in config["model"].items():
        setattr(args, key, value)
    setattr(args, "max_train_steps", config["training"].get("max_train_steps"))
    setattr(args, "max_test_steps", config["training"].get("max_test_steps"))

    # Initialize directories
    _init_(args, config)

    # Start training
    train(args, config)


if __name__ == "__main__":
    main()
