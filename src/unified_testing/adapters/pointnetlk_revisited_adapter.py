"""
PointNetLK_Revisited (CVPR 2021) algorithm adapter

Supports voxelization and multiple datasets (ModelNet40, 3DMatch, ShapeNet, KITTI)
"""

import importlib.util
import os
import sys

import numpy as np
import torch

from ..core.base_adapter import BaseAdapter

# add PointNetLK_Revisited path
POINTNETLK_REVISITED_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "baselines", "PointNetLK_Revisited"
    )
)


def _load_pointnetlk_revisited_modules():
    """Load PointNetLK_Revisited modules without polluting global import state."""
    utils_path = os.path.join(POINTNETLK_REVISITED_PATH, "utils.py")
    model_path = os.path.join(POINTNETLK_REVISITED_PATH, "model.py")
    original_utils_module = sys.modules.get("utils")

    try:
        utils_spec = importlib.util.spec_from_file_location(
            "pointnetlk_revisited_utils", utils_path
        )
        if utils_spec is None or utils_spec.loader is None:
            raise ImportError(f"Unable to load spec for {utils_path}")
        plk_utils = importlib.util.module_from_spec(utils_spec)
        sys.modules["pointnetlk_revisited_utils"] = plk_utils
        sys.modules["utils"] = plk_utils
        utils_spec.loader.exec_module(plk_utils)

        model_spec = importlib.util.spec_from_file_location(
            "pointnetlk_revisited_model", model_path
        )
        if model_spec is None or model_spec.loader is None:
            raise ImportError(f"Unable to load spec for {model_path}")
        plk_model = importlib.util.module_from_spec(model_spec)
        sys.modules["pointnetlk_revisited_model"] = plk_model
        model_spec.loader.exec_module(plk_model)
    finally:
        if original_utils_module is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = original_utils_module

    return plk_model.Pointnet_Features, plk_model.AnalyticalPointNetLK


try:
    Pointnet_Features, AnalyticalPointNetLK = _load_pointnetlk_revisited_modules()
except ImportError as e:
    print(f"Error importing PointNetLK_Revisited modules: {e}")
    print(
        f"Make sure PointNetLK_Revisited is available at: {POINTNETLK_REVISITED_PATH}"
    )
    raise


class PointNetLKRevisitedAdapter(BaseAdapter):
    """PointNetLK_Revisited adapter with voxelization support"""

    def __init__(self, args):
        super().__init__(args)

        # get PointLK parameters
        self.max_iter = getattr(args, "max_iter", 10)
        self.xtol = getattr(args, "xtol", 1.0e-7)
        self.dim_k = getattr(args, "dim_k", 1024)

        # voxelization parameters (optional)
        self.use_voxelization = getattr(args, "use_voxelization", False)
        self.voxel_size = getattr(args, "voxel_size", 0.02)

        # data type
        self.data_type = getattr(
            args, "data_type", "synthetic"
        )  # 'synthetic' or 'real'

        # Normalization mode
        # - 'none': no normalization
        # - 'unit_cube': apply target/template unit-cube transform to both clouds
        # - 'joint': joint normalization (source and target together)
        self.normalize_mode = getattr(args, "normalize_mode", "none")

        print("PointNetLK_Revisited adapter initialized:")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Xtol: {self.xtol}")
        print(f"  Dim k: {self.dim_k}")
        print(f"  Use voxelization: {self.use_voxelization}")
        print(f"  Data type: {self.data_type}")
        print(f"  Normalization mode: {self.normalize_mode}")

        # PointNetLK family baselines expect perturbations to be valid in the
        # normalized coordinate frame as well.
        self._supports_normalized_perturbation = self.normalize_mode in [
            "joint",
            "unit_cube",
        ]

    def supports_normalized_perturbation(self):
        """
        Check if this adapter supports applying perturbations in normalized space.

        PointNetLK-family baselines can consume perturbations in normalized space:
        normalize → perturb → predict.

        Returns:
            bool: True if perturbations should be applied after normalization
        """
        return self._supports_normalized_perturbation

    def load_model(self, model_path):
        """
        Load PointNetLK_Revisited model

        Args:
            model_path: path to PointNetLK checkpoint (.pth file)
        """
        # create feature extractor
        ptnet = Pointnet_Features(dim_k=self.dim_k)

        # create PointNetLK model
        self.model = AnalyticalPointNetLK(ptnet, self.device)

        # load weights
        if model_path and os.path.exists(model_path):
            print(f"Loading PointNetLK_Revisited model from: {model_path}")
            checkpoint = torch.load(model_path, map_location="cpu")

            # handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Use strict=False to allow partial loading
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )
            print("✓ Loaded PointNetLK_Revisited model")
            if missing_keys:
                print(
                    "  Warning: Missing keys (will use random init): "
                    f"{len(missing_keys)} keys"
                )
                print(f"    First few: {missing_keys[:3]}")
            if unexpected_keys:
                print(
                    "  Warning: Unexpected keys in checkpoint: "
                    f"{len(unexpected_keys)} keys"
                )
        else:
            print(f"Warning: Model path not found or not specified: {model_path}")
            print("Using randomly initialized weights")

        # move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, source, target):
        """
        Preprocess point clouds for PointNetLK_Revisited

        PointNetLK_Revisited expects: [B, N, 3] format

        Args:
            source: [N, 3] numpy array (perturbed source)
            target: [N, 3] numpy array (target/template)

        Returns:
            source_tensor: [1, N, 3] torch.Tensor
            target_tensor: [1, N, 3] torch.Tensor
            voxel_coords_source: voxel coordinates (for voxelization mode)
            voxel_coords_target: voxel coordinates (for voxelization mode)
        """
        source_norm, target_norm, _ = self._normalize_pair(source, target)

        # convert to tensor and add batch dimension
        source_tensor = torch.from_numpy(source_norm).float().unsqueeze(0)  # [1, N, 3]
        target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0)  # [1, N, 3]

        # move to device
        # PointNetLK_Revisited requires gradient computation for analytical Jacobian.
        # Do NOT call .detach() or .requires_grad_(False) here!
        source_tensor = source_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # compute voxel coordinates (for real data mode)
        if self.data_type == "real":
            # compute voxel center coordinates
            voxel_coords_source = source_tensor.mean(dim=1)  # [1, 3]
            voxel_coords_target = target_tensor.mean(dim=1)  # [1, 3]
        else:
            # synthetic data: use None
            voxel_coords_source = None
            voxel_coords_target = None

        return source_tensor, target_tensor, voxel_coords_source, voxel_coords_target

    def forward(
        self,
        source_tensor,
        target_tensor,
        voxel_coords_source=None,
        voxel_coords_target=None,
    ):
        """
        Forward inference through PointNetLK_Revisited

        Args:
            source_tensor: [1, N, 3] perturbed source
            target_tensor: [1, N, 3] template (target)
            voxel_coords_source: voxel coordinates for source (optional)
            voxel_coords_target: voxel coordinates for target (optional)

        Returns:
            g_pred: [1, 4, 4] predicted SE(3) transformation
        """
        # PointNetLK_Revisited requires gradient computation for analytical Jacobian.
        # Use set_grad_enabled(True) to allow gradient computation without history.
        with torch.set_grad_enabled(True):
            # AnalyticalPointNetLK iterative refinement
            # Note: expects (template, source) order
            _ = AnalyticalPointNetLK.do_forward(
                self.model,
                target_tensor,  # p0 (template)
                voxel_coords_target
                if voxel_coords_target is not None
                else target_tensor.mean(dim=1),
                source_tensor,  # p1 (source)
                voxel_coords_source
                if voxel_coords_source is not None
                else source_tensor.mean(dim=1),
                maxiter=self.max_iter,
                xtol=self.xtol,
                p0_zero_mean=True,
                p1_zero_mean=True,
                mode="test",
                data_type=self.data_type,
                num_random_points=100,
            )

            # get predicted transformation (stored in model.g)
            g_pred = (
                self.model.g.detach()
            )  # [1, 4, 4] - detach output to avoid memory leak

        return g_pred

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

    def preprocess_for_perturbation(self, source, target):
        """
        Preprocess point clouds for perturbation application.

        This method applies normalization WITHOUT applying any perturbation,
        so that the perturbation can be applied in the normalized space.

        Args:
            source: [N, 3] numpy array (original source)
            target: [N, 3] numpy array (original target)

        Returns:
            source_norm: [N, 3] numpy array (normalized source)
            target_norm: [N, 3] numpy array (normalized target)
            preprocess_info: dict with normalization parameters
        """
        source_norm, target_norm, preprocess_info = self._normalize_pair(source, target)

        return source_norm, target_norm, preprocess_info

    def predict_after_perturbation(
        self, source_perturbed, target_norm, preprocess_info
    ):
        """
        Predict transformation after perturbation has been applied in normalized space.

        This method assumes that:
        1. source and target have already been normalized
        2. Perturbation has been applied to source in the normalized space
        3. No additional preprocessing is needed

        Args:
            source_perturbed: [N, 3] numpy array (perturbed source in normalized space)
            target_norm: [N, 3] numpy array (normalized target)
            preprocess_info: dict with normalization parameters

        Returns:
            R: [3, 3] rotation matrix
            t: [3,] translation vector
        """
        # Set to eval mode
        self.set_eval_mode()

        # Convert to tensors (no additional normalization needed)
        source_tensor = (
            torch.from_numpy(source_perturbed).float().unsqueeze(0)
        )  # [1, N, 3]
        target_tensor = torch.from_numpy(target_norm).float().unsqueeze(0)  # [1, N, 3]

        # Move to device
        source_tensor = source_tensor.to(self.device)
        target_tensor = target_tensor.to(self.device)

        # Compute voxel coordinates (for real data mode)
        if self.data_type == "real":
            voxel_coords_source = source_tensor.mean(dim=1)  # [1, 3]
            voxel_coords_target = target_tensor.mean(dim=1)  # [1, 3]
        else:
            voxel_coords_source = None
            voxel_coords_target = None

        # Forward pass
        output = self.forward(
            source_tensor, target_tensor, voxel_coords_source, voxel_coords_target
        )

        # Extract transformation
        g_pred = output.detach().cpu().numpy()[0]  # [4, 4]
        R = g_pred[:3, :3]
        t = g_pred[:3, 3]

        # IMPORTANT: Return translation in NORMALIZED space
        # Do NOT denormalize translation here. Error computation should be done
        # in the same space (normalized space) to ensure consistency.
        # The old code incorrectly denormalized t, causing mixed-unit error calculation.

        return R, t

    def predict(self, source, target):
        """
        Override predict to handle voxel coordinates

        Args:
            source: [N, 3] numpy array
            target: [N, 3] numpy array

        Returns:
            R: [3, 3] rotation matrix
            t: [3,] translation vector
        """
        # set to eval mode
        self.set_eval_mode()

        # preprocess
        source_tensor, target_tensor, voxel_coords_source, voxel_coords_target = (
            self.preprocess(source, target)
        )

        # forward
        output = self.forward(
            source_tensor, target_tensor, voxel_coords_source, voxel_coords_target
        )

        # extract transformation
        R, t = self.extract_transformation(output)

        return R, t

    def get_data_format(self):
        """Return data format"""
        return "BN3"  # [B, N, 3]

    def get_algorithm_name(self):
        """Return algorithm name"""
        return "PointNetLK_Revisited"
