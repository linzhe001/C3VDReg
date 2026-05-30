"""
ICP (Iterative Closest Point) adapter for unified testing framework

ICP is a classic point cloud registration algorithm that iteratively:
1. Finds nearest neighbors between source and target
2. Computes rigid transformation using SVD
3. Applies transformation and repeats

No neural network - pure geometric algorithm
"""

import numpy as np
import torch
from scipy.spatial import KDTree

from src.common.utils.benchmark_preprocess import (
    normalize_point_cloud_pair,
    recover_raw_transform,
)


class ICPAdapter:
    """Adapter for ICP algorithm"""

    def __init__(self, config):
        """
        Initialize ICP adapter

        Args:
            config: dict or Namespace with configuration
                max_iter: maximum iterations (default: 20)
                ftol: convergence tolerance (default: 1e-7)
                device: 'cuda' or 'cpu' (not used, ICP is CPU-based)
        """
        # Convert Namespace to dict if needed
        if hasattr(config, "__dict__"):
            config = vars(config)

        self.max_iter = config.get("max_iter", 20)
        self.ftol = config.get("ftol", 1e-7)
        self.device = config.get("device", "cpu")

        self.normalize_mode = config.get("normalize_mode", "unit_cube")
        self._source_normalization_transform = np.eye(4, dtype=np.float64)
        self._target_normalization_transform = np.eye(4, dtype=np.float64)

        print(f"ICP Configuration (Correct Workflow Version):")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Convergence tolerance: {self.ftol}")
        print(f"  Normalization mode: {self.normalize_mode}")
        print(f"  Note: ICP is CPU-based algorithm (device setting ignored)")

    def supports_normalized_perturbation(self):
        """Check if this adapter supports applying perturbations in normalized space."""
        return self.normalize_mode in {"joint", "unit_cube"}

    def _set_normalization_transforms(
        self,
        source_transform: np.ndarray | None = None,
        target_transform: np.ndarray | None = None,
    ) -> None:
        self._source_normalization_transform = (
            np.eye(4, dtype=np.float64)
            if source_transform is None
            else np.asarray(source_transform, dtype=np.float64)
        )
        self._target_normalization_transform = (
            np.eye(4, dtype=np.float64)
            if target_transform is None
            else np.asarray(target_transform, dtype=np.float64)
        )

    def _normalize_pair(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        source_norm, target_norm, info, source_transform, target_transform = (
            normalize_point_cloud_pair(source, target, self.normalize_mode)
        )

        self._set_normalization_transforms(source_transform, target_transform)
        info["source_norm_transform"] = source_transform.tolist()
        info["target_norm_transform"] = target_transform.tolist()
        return source_norm, target_norm, info

    def _recover_raw_transform(self, normalized_transform: np.ndarray) -> np.ndarray:
        return recover_raw_transform(
            normalized_transform=normalized_transform,
            source_norm_transform=self._source_normalization_transform,
            target_norm_transform=self._target_normalization_transform,
        )

    def load_model(self, checkpoint_path=None):
        """
        Load model (no-op for ICP - no weights to load)

        Args:
            checkpoint_path: path to checkpoint (ignored for ICP)
        """
        if checkpoint_path:
            print("Warning: ICP has no model weights to load (checkpoint path ignored)")
        print("✓ ICP ready (no model loading required)")

    def forward(self, source, target):
        """
        Perform ICP registration

        Args:
            source: source point cloud [N, 3] or [B, N, 3]
            target: target point cloud [M, 3] or [B, M, 3]

        Returns:
            predicted_transform: [4, 4] or [B, 4, 4] transformation matrix
        """
        # Handle batched input
        if len(source.shape) == 3:
            batch_size = source.shape[0]
            transforms = []
            for i in range(batch_size):
                T = self._icp_single(source[i], target[i])
                transforms.append(T)
            return torch.stack(transforms, dim=0)
        else:
            # Single pair
            return self._icp_single(source, target)

    def _icp_single(self, source, target):
        """
        ICP for single point cloud pair

        Note: ICP estimates transformation such that target = T @ source
        where T transforms source to match target (p0 = g.p1 in PointNetLK notation)

        Args:
            source: source point cloud [N, 3] (p1 in PointNetLK)
            target: target point cloud [M, 3] (p0 in PointNetLK)

        Returns:
            transform: [4, 4] transformation matrix (g such that p0 = g.p1)
        """
        # Convert to numpy if needed
        if isinstance(source, torch.Tensor):
            p1_np = source.cpu().numpy()  # p1: source
        else:
            p1_np = np.array(source)

        if isinstance(target, torch.Tensor):
            p0_np = target.cpu().numpy()  # p0: target
        else:
            p0_np = np.array(target)

        # Ensure float64 for numerical stability
        p1_np = p1_np.astype(np.float64)
        p0_np = p0_np.astype(np.float64)

        # Build KDTree for target (p0)
        tree = KDTree(p0_np, leafsize=1000)

        # Initialize transformation: g such that p0 = g.p1
        g = np.eye(4, dtype=np.float64)
        p = np.copy(p1_np)  # Current transformed source

        # ICP iterations
        for itr in range(self.max_iter):
            # Find nearest neighbors in target (p0)
            _, neighbor_idx = tree.query(p)
            targets = p0_np[neighbor_idx]

            # Compute transformation from current p to targets
            R, t = self._find_rigid_transform(p, targets)

            # Apply transformation
            new_p = np.dot(R, p.T).T + t

            # Check convergence
            if np.sum(np.abs(p - new_p)) < self.ftol:
                break

            # Update
            p = np.copy(new_p)
            dg = self._Rt_to_matrix(R, t)
            g = np.dot(dg, g)

        # Convert to torch tensor
        transform = torch.from_numpy(g).float()

        return transform

    def _find_rigid_transform(self, p_from, p_target):
        """
        Find rigid transformation (R, t) such that: p_target = R * p_from + t

        Uses SVD-based method (Arun et al. 1987)

        Args:
            p_from: [N, 3] source points
            p_target: [N, 3] target points

        Returns:
            R: [3, 3] rotation matrix
            t: [3] translation vector
        """
        A = np.copy(p_from)
        B = np.copy(p_target)

        # Compute centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        # Center the points
        A -= centroid_A
        B -= centroid_B

        # Compute rotation using SVD
        H = np.dot(A.T, B)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Compute translation
        t = np.dot(-R, centroid_A) + centroid_B

        return R, t

    def _Rt_to_matrix(self, R, t):
        """
        Convert (R, t) to 4x4 transformation matrix

        Args:
            R: [3, 3] rotation matrix
            t: [3] translation vector

        Returns:
            M: [4, 4] transformation matrix
        """
        # M = [R, t; 0, 1]
        Rt = np.concatenate((R, np.expand_dims(t.T, axis=-1)), axis=1)
        bottom = np.concatenate((np.zeros(3), np.ones(1)))
        M = np.concatenate((Rt, np.expand_dims(bottom, axis=0)), axis=0)
        return M

    def predict(self, source, target):
        """
        Unified interface for prediction (required by unified_test.py)

        Args:
            source: [N, 3] numpy array (perturbed source)
            target: [M, 3] numpy array (target)

        Returns:
            R: [3, 3] rotation matrix (numpy)
            t: [3] translation vector (numpy)
        """
        source_norm, target_norm, _ = self._normalize_pair(source, target)
        g_pred = self._icp_single(source_norm, target_norm)
        if isinstance(g_pred, torch.Tensor):
            g_pred = g_pred.cpu().numpy()
        g_pred = self._recover_raw_transform(g_pred)

        # Extract R and t from 4x4 matrix (convert to numpy if needed)
        R = g_pred[:3, :3]
        t = g_pred[:3, 3]

        # Ensure numpy arrays are returned
        if isinstance(R, torch.Tensor):
            R = R.cpu().numpy()
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()

        return R, t

    def preprocess_for_perturbation(self, source, target):
        """
        Preprocess point clouds for perturbation.
        ICP CORRECT WORKFLOW:
        1. Normalize point clouds (to get scale and center)
        2. Return normalized points and normalization info

        Args:
            source: [N, 3] source point cloud (numpy)
            target: [M, 3] target point cloud (numpy)

        Returns:
            source_norm: [N, 3] normalized source (numpy)
            target_norm: [M, 3] normalized target (numpy)
            info: dict with 'center' and 'scale' for later denormalization
        """
        return self._normalize_pair(source, target)

    def predict_after_perturbation(self, source_perturbed, target, info):
        """
        Predict transformation using perturbed point clouds.
        ICP CORRECT WORKFLOW:
        1. Denormalize perturbed source and target back to raw space
        2. Run ICP in raw space
        3. Convert prediction back to normalized space

        Args:
            source_perturbed: [N, 3] perturbed source in NORMALIZED space (numpy)
            target: [M, 3] target in NORMALIZED space (numpy)
            info: dict with 'center' and 'scale' from preprocessing

        Returns:
            R: [3, 3] rotation matrix in NORMALIZED space (numpy)
            t: [3,] translation vector in NORMALIZED space (numpy)
        """
        _ = info
        g_pred = self._icp_single(source_perturbed, target)
        if isinstance(g_pred, torch.Tensor):
            g_pred = g_pred.cpu().numpy()
        return g_pred[:3, :3], g_pred[:3, 3]

    def get_algorithm_name(self):
        """Get algorithm name for reporting"""
        return "ICP"

    def eval(self):
        """Set to eval mode (no-op for ICP)"""
        pass

    def train(self):
        """Set to train mode (no-op for ICP)"""
        pass

    def __repr__(self):
        return f"ICPAdapter(max_iter={self.max_iter}, ftol={self.ftol})"


# Example usage and testing
if __name__ == "__main__":
    print("Testing ICP Adapter...")

    # Create test data
    from math import sin, cos

    # Generate source points
    Y, X = np.mgrid[0:10:1, 0:10:1]
    Z = Y**2 + X**2
    source = np.vstack([Y.reshape(-1), X.reshape(-1), Z.reshape(-1)]).T
    source = source.astype(np.float32)

    # Apply known transformation
    angle = 0.279
    R = np.array(
        [[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    t = np.array([5.0, 20.0, 10.0], dtype=np.float32)

    target = np.dot(R, source.T).T + t

    # Test ICP
    config = {"max_iter": 20, "ftol": 1e-7, "device": "cpu"}

    icp = ICPAdapter(config)
    icp.load_model()

    # Convert to torch
    source_torch = torch.from_numpy(source)
    target_torch = torch.from_numpy(target)

    # Run ICP
    print("\nRunning ICP...")
    transform = icp.forward(source_torch, target_torch)

    print(f"\nEstimated transformation:\n{transform}")

    # Expected: inverse transformation
    R_inv = R.T
    t_inv = -np.dot(R_inv, t)
    expected = np.eye(4, dtype=np.float32)
    expected[:3, :3] = R_inv
    expected[:3, 3] = t_inv

    print(f"\nExpected transformation:\n{expected}")

    # Check error
    error = np.abs(transform.numpy() - expected).max()
    print(f"\nMax error: {error:.6f}")

    if error < 1e-3:
        print("✓ ICP test PASSED")
    else:
        print("✗ ICP test FAILED")
