"""
PointNetLK (Original CVPR 2019) algorithm adapter

For ModelNet40 dataset
"""

import sys
import os
import torch
import numpy as np
import importlib.util

# CRITICAL: Explicitly load modules from original PointNetLK directory
# to avoid conflicts with PointNetLK_c3vd
POINTNETLK_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "baselines", "PointNetLK")
)
PTLK_DIR = os.path.join(POINTNETLK_DIR, "ptlk")

# Add PointNetLK to path temporarily for imports
if POINTNETLK_DIR not in sys.path:
    sys.path.insert(0, POINTNETLK_DIR)

# Import se3 and so3 modules first (they are needed by pointlk)
sys.path.insert(0, PTLK_DIR)

try:
    # Load se3 module
    se3_path = os.path.join(PTLK_DIR, "se3.py")
    spec = importlib.util.spec_from_file_location("ptlk.se3", se3_path)
    se3 = importlib.util.module_from_spec(spec)
    se3.__package__ = "ptlk"
    sys.modules["ptlk.se3"] = se3
    spec.loader.exec_module(se3)

    # Load so3 module
    so3_path = os.path.join(PTLK_DIR, "so3.py")
    spec = importlib.util.spec_from_file_location("ptlk.so3", so3_path)
    so3 = importlib.util.module_from_spec(spec)
    so3.__package__ = "ptlk"
    sys.modules["ptlk.so3"] = so3
    spec.loader.exec_module(so3)

    # Load pointnet module
    pointnet_path = os.path.join(PTLK_DIR, "pointnet.py")
    spec = importlib.util.spec_from_file_location("ptlk.pointnet", pointnet_path)
    pointnet = importlib.util.module_from_spec(spec)
    pointnet.__package__ = "ptlk"
    sys.modules["ptlk.pointnet"] = pointnet
    spec.loader.exec_module(pointnet)

    # Load pointlk module (it will now find pointnet via sys.modules)
    pointlk_path = os.path.join(PTLK_DIR, "pointlk.py")
    spec = importlib.util.spec_from_file_location("ptlk.pointlk", pointlk_path)
    pointlk_module = importlib.util.module_from_spec(spec)
    pointlk_module.__package__ = "ptlk"
    sys.modules["ptlk.pointlk"] = pointlk_module
    spec.loader.exec_module(pointlk_module)
    pointlk = pointlk_module

    # Verify we loaded the correct version
    import inspect

    sig = inspect.signature(pointnet.PointNet_features.__init__)
    if "use_tnet" not in sig.parameters:
        raise ImportError(
            f"Wrong pointnet module! Expected use_tnet parameter. Module: {pointnet_path}"
        )

    print(f"✓ Loaded PointNetLK modules from: {POINTNETLK_DIR}")

finally:
    # Clean up path
    if PTLK_DIR in sys.path:
        sys.path.remove(PTLK_DIR)

from ..core.base_adapter import BaseAdapter


class PointNetLKAdapter(BaseAdapter):
    """PointNetLK (Original) adapter for ModelNet40"""

    def __init__(self, args):
        super().__init__(args)

        # get PointLK parameters
        self.max_iter = getattr(args, "max_iter", 20)
        self.delta = getattr(args, "delta", 1.0e-2)
        self.dim_k = getattr(args, "dim_k", 1024)
        self.use_tnet = getattr(args, "use_tnet", True)
        self.symfn = getattr(args, "symfn", "max")  # 'max' or 'avg'

        # Normalization mode
        # - 'none': no normalization
        # - 'unit_cube': apply target/template unit-cube transform to both clouds
        # - 'joint': joint normalization (source and target together)
        self.normalize_mode = getattr(args, "normalize_mode", "unit_cube")

        print(f"PointNetLK adapter initialized:")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Delta: {self.delta}")
        print(f"  Dim k: {self.dim_k}")
        print(f"  Use TNet: {self.use_tnet}")
        print(f"  Symmetric function: {self.symfn}")
        print(f"  Normalization mode: {self.normalize_mode}")

    def supports_normalized_perturbation(self):
        """Check if this adapter supports applying perturbations in normalized space."""
        # Enable normalized perturbation when using joint or unit_cube normalization
        return self.normalize_mode in ["joint", "unit_cube"]

    def load_model(self, model_path):
        """
        Load PointNetLK model

        Args:
            model_path: path to PointNetLK checkpoint (.pth file)
        """
        # select symmetric function
        if self.symfn == "max":
            sym_fn = pointnet.symfn_max
        elif self.symfn == "avg":
            sym_fn = pointnet.symfn_avg
        else:
            raise ValueError(f"Unknown symmetric function: {self.symfn}")

        # create feature extractor
        self.model = pointnet.PointNet_features(
            dim_k=self.dim_k, use_tnet=self.use_tnet, sym_fn=sym_fn
        )

        # load classifier weights (if specified for transfer learning)
        transfer_from = getattr(self.args, "transfer_from", None)
        if transfer_from and os.path.exists(transfer_from):
            print(f"Loading classifier features from: {transfer_from}")
            state_dict = torch.load(transfer_from, map_location="cpu")

            # handle different checkpoint formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            self.model.load_state_dict(state_dict, strict=False)
            print("✓ Loaded classifier features")

        # load PointNetLK weights
        if model_path and os.path.exists(model_path):
            print(f"Loading PointNetLK model from: {model_path}")
            state_dict = torch.load(model_path, map_location="cpu")

            # handle different checkpoint formats
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Handle PointLK wrapper checkpoint (with 'ptnet.' prefix)
            if any(k.startswith("ptnet.") for k in state_dict.keys()):
                print(
                    "  Detected PointLK wrapper checkpoint, extracting PointNet weights..."
                )
                # Extract only ptnet weights and remove prefix
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("ptnet."):
                        new_key = k[6:]  # Remove 'ptnet.' prefix
                        new_state_dict[new_key] = v
                state_dict = new_state_dict

            self.model.load_state_dict(state_dict, strict=False)
            print("✓ Loaded PointNetLK model")
        else:
            print(f"Warning: Model path not found or not specified: {model_path}")
            print("Using randomly initialized weights")

        # move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

        # Create PointLK wrapper for iterative refinement
        self.pointlk = pointlk.PointLK(ptnet=self.model, delta=self.delta)
        self.pointlk.to(self.device)
        self.pointlk.eval()

    def preprocess(self, source, target):
        """
        Preprocess point clouds for PointNetLK

        PointNetLK expects: [B, N, 3] format

        Args:
            source: [N, 3] numpy array (perturbed source)
            target: [N, 3] numpy array (target/template)

        Returns:
            source_tensor: [1, N, 3] torch.Tensor
            target_tensor: [1, N, 3] torch.Tensor
        """
        source_norm, target_norm, _ = self._normalize_pair(source, target)

        # convert to tensor and add batch dimension
        source_tensor = torch.from_numpy(source_norm).float().unsqueeze(0)  # [1, N, 3]
        target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0)  # [1, N, 3]

        # move to device
        source_tensor = source_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        return source_tensor, target_tensor

    def forward(self, source_tensor, target_tensor):
        """
        Forward inference through PointNetLK

        Args:
            source_tensor: [1, N, 3] perturbed source
            target_tensor: [1, N, 3] template (target)

        Returns:
            g_pred: [1, 4, 4] predicted SE(3) transformation
        """
        with torch.no_grad():
            # PointNetLK iterative refinement
            # Note: PointLK expects (template, source) order
            _ = pointlk.PointLK.do_forward(
                self.pointlk,
                target_tensor,  # p0 (template)
                source_tensor,  # p1 (source)
                maxiter=self.max_iter,
                xtol=1e-7,
                p0_zero_mean=True,
                p1_zero_mean=True,
            )

            # get predicted transformation (stored in pointlk.g)
            g_pred = self.pointlk.g  # [1, 4, 4]

        return g_pred

    def preprocess_for_perturbation(self, source, target):
        """
        Preprocess point clouds for perturbation in normalized space.
        This normalizes the point clouds but does NOT apply perturbation.

        Args:
            source: [N, 3] source point cloud (numpy)
            target: [M, 3] target point cloud (numpy)

        Returns:
            source_norm: [1, N, 3] normalized source (torch tensor)
            target_norm: [1, M, 3] normalized target (torch tensor)
            info: dict with normalization metadata
        """
        source_norm, target_norm, info = self._normalize_pair(source, target)

        # Convert to tensor
        source_tensor = (
            torch.from_numpy(source_norm).float().unsqueeze(0).to(self.device)
        )
        target_tensor = (
            torch.from_numpy(target_norm).float().unsqueeze(0).to(self.device)
        )

        return source_tensor, target_tensor, info

    def predict_after_perturbation(self, source_perturbed, target_norm, info):
        """
        Predict transformation using already-perturbed normalized point clouds.

        Args:
            source_perturbed: [1, N, 3] perturbed source in normalized space
            target_norm: [1, M, 3] normalized target
            info: dict with normalization metadata

        Returns:
            R: [3, 3] rotation matrix (numpy)
            t: [3,] translation vector in original space (numpy)
        """
        with torch.no_grad():
            # PointNetLK iterative refinement
            _ = pointlk.PointLK.do_forward(
                self.pointlk,
                target_norm,  # p0 (template)
                source_perturbed,  # p1 (source)
                maxiter=self.max_iter,
                xtol=1e-7,
                p0_zero_mean=True,
                p1_zero_mean=True,
            )

            g_pred = self.pointlk.g  # [1, 4, 4]

        # Extract R and t
        g_pred_np = g_pred.detach().cpu().numpy()[0]  # [4, 4]
        R = g_pred_np[:3, :3]
        t = g_pred_np[:3, 3]

        # IMPORTANT: Return translation in NORMALIZED space
        # Do NOT denormalize translation here. Error computation should be done
        # in the same space (normalized space) to ensure consistency.
        # The old code incorrectly denormalized t, causing mixed-unit error calculation.

        return R, t

    def extract_transformation(self, output):
        """
        Extract rotation and translation from SE(3) matrix

        Args:
            output: [1, 4, 4] SE(3) matrix

        Returns:
            R: [3, 3] rotation matrix (numpy)
            t: [3,] translation vector (numpy)
        """
        g_pred = output.detach().cpu().numpy()[0]  # [4, 4]

        R = g_pred[:3, :3]
        t_norm = g_pred[:3, 3]

        transform_norm = np.eye(4, dtype=np.float64)
        transform_norm[:3, :3] = R
        transform_norm[:3, 3] = t_norm
        transform_raw = self._recover_raw_transform(transform_norm)
        t = transform_raw[:3, 3]
        R = transform_raw[:3, :3]

        return R, t

    def get_data_format(self):
        """Return data format"""
        return "BN3"  # [B, N, 3]

    def get_algorithm_name(self):
        """Return algorithm name"""
        return "PointNetLK"
