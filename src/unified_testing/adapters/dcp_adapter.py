"""
DCP (Deep Closest Point) algorithm adapter
"""

import sys
import os
import torch
import numpy as np

# add DCP path
DCP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "dcp")
)
if DCP_PATH not in sys.path:
    sys.path.insert(0, DCP_PATH)

# Save original sys.path state to avoid conflicts
import importlib.util

try:
    # Load DCP model module specifically
    model_spec = importlib.util.spec_from_file_location(
        "dcp_model", os.path.join(DCP_PATH, "model.py")
    )
    dcp_model = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(dcp_model)
    DCP = dcp_model.DCP
except Exception as e:
    print(f"Error importing DCP modules: {e}")
    print(f"Make sure DCP is available at: {DCP_PATH}")
    raise

from ..core.base_adapter import BaseAdapter


class DCPAdapter(BaseAdapter):
    """DCP algorithm adapter"""

    def __init__(self, args):
        super().__init__(args)

        # DCP parameters
        self.emb_nn = getattr(args, "emb_nn", "dgcnn")
        self.pointer = getattr(args, "pointer", "transformer")
        self.head = getattr(args, "head", "svd")
        self.emb_dims = getattr(args, "emb_dims", 512)
        self.n_blocks = getattr(args, "n_blocks", 1)
        self.n_heads = getattr(args, "n_heads", 4)
        self.ff_dims = getattr(args, "ff_dims", 1024)
        self.dropout = getattr(args, "dropout", 0.0)

        # Normalization mode
        # - 'none': no normalization
        # - 'unit_cube': apply target/template unit-cube transform to both clouds
        # - 'joint': joint normalization (source and target together)
        self.normalize_mode = getattr(args, "normalize_mode", "unit_cube")

        print(f"DCP adapter initialized:")
        print(f"  Embedding: {self.emb_nn}")
        print(f"  Normalization mode: {self.normalize_mode}")
        print(f"  Pointer: {self.pointer}")
        print(f"  Head: {self.head}")
        print(f"  Emb dims: {self.emb_dims}")

        # Enable normalized-space perturbation for models trained with normalization
        self._supports_normalized_perturbation = self.normalize_mode in [
            "joint",
            "unit_cube",
        ]

    def supports_normalized_perturbation(self):
        """Check if this adapter supports applying perturbations in normalized space."""
        return self._supports_normalized_perturbation

    def load_model(self, model_path):
        """
        Load DCP model

        Args:
            model_path: path to DCP checkpoint (.t7 or .pth file)
        """
        # create DCP model
        self.model = DCP(self.args)

        # load weights
        if model_path and os.path.exists(model_path):
            print(f"Loading DCP model from: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")

            # handle different checkpoint formats
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Use strict=False to handle different model versions
            # Pretrained models may have different BatchNorm structure
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                print(
                    f"  Warning: Missing keys in checkpoint: {len(missing_keys)} keys"
                )
            if unexpected_keys:
                print(
                    f"  Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys"
                )

            print("✓ Loaded DCP model (with strict=False)")
        else:
            print(f"Warning: Model path not found: {model_path}")
            print("Using randomly initialized weights")

        # move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, source, target):
        """
        Preprocess point clouds for DCP

        DCP consumes B3N tensors after any benchmark-selected normalization policy.

        Args:
            source: [N, 3] numpy array
            target: [N, 3] numpy array

        Returns:
            source_tensor: [1, 3, N] torch.Tensor
            target_tensor: [1, 3, N] torch.Tensor
        """
        source_norm, target_norm, _ = self._normalize_pair(source, target)

        source_tensor = torch.from_numpy(source_norm.T).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_norm.T).float().unsqueeze(0)

        # move to device
        source_tensor = source_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        return source_tensor, target_tensor

    def preprocess_for_perturbation(self, source, target):
        """
        Preprocess point clouds for perturbation application (normalize without perturbation).

        Args:
            source: [N, 3] numpy array
            target: [N, 3] numpy array

        Returns:
            source_norm: [N, 3] numpy array (normalized)
            target_norm: [N, 3] numpy array (normalized)
            preprocess_info: dict with normalization parameters
        """
        source_norm, target_norm, preprocess_info = self._normalize_pair(source, target)

        return source_norm, target_norm, preprocess_info

    def predict_after_perturbation(
        self, source_perturbed, target_norm, preprocess_info
    ):
        """
        Predict transformation after perturbation in normalized space.

        Args:
            source_perturbed: [N, 3] numpy array (perturbed in normalized space)
            target_norm: [N, 3] numpy array (normalized target)
            preprocess_info: dict with normalization parameters

        Returns:
            R: [3, 3] rotation matrix
            t: [3,] translation vector
        """
        self.set_eval_mode()

        # Convert to tensors in DCP format [1, 3, N]
        source_tensor = torch.from_numpy(source_perturbed.T).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_norm.T).float().unsqueeze(0)

        # Move to device
        source_tensor = source_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # Forward pass
        output = self.forward(source_tensor, target_tensor)

        # Extract transformation
        rotation, translation = output
        R = rotation.detach().cpu().numpy()[0]  # [3, 3]
        t = translation.detach().cpu().numpy()[0]  # [3,]

        # IMPORTANT: Return translation in NORMALIZED space
        # Do NOT denormalize translation here. Error computation should be done
        # in the same space (normalized space) to ensure consistency.
        # The old code incorrectly denormalized t, causing mixed-unit error calculation.

        return R, t

    def forward(self, source_tensor, target_tensor):
        """
        Forward inference through DCP

        Args:
            source_tensor: [1, 3, N]
            target_tensor: [1, 3, N]

        Returns:
            rotation: [1, 3, 3] rotation matrix
            translation: [1, 3] translation vector
        """
        with torch.no_grad():
            # DCP forward pass
            # Returns: rotation, translation, src_embedding, tgt_embedding
            rotation, translation, _, _ = self.model(source_tensor, target_tensor)

        return rotation, translation

    def extract_transformation(self, output):
        """
        Extract rotation and translation from DCP output

        Args:
            output: tuple (rotation, translation)
                rotation: [1, 3, 3] tensor
                translation: [1, 3] tensor

        Returns:
            R: [3, 3] rotation matrix (numpy)
            t: [3,] translation vector (numpy)
        """
        rotation, translation = output

        R = rotation.detach().cpu().numpy()[0]  # [3, 3]
        t_norm = translation.detach().cpu().numpy()[0]  # [3,] in normalized space

        transform_norm = np.eye(4, dtype=np.float64)
        transform_norm[:3, :3] = R
        transform_norm[:3, 3] = t_norm
        transform_raw = self._recover_raw_transform(transform_norm)
        t = transform_raw[:3, 3]
        R = transform_raw[:3, :3]

        return R, t

    def get_data_format(self):
        """Return data format"""
        return "B3N"  # [B, 3, N]

    def get_algorithm_name(self):
        """Return algorithm name"""
        return f"DCP_{self.pointer}"
