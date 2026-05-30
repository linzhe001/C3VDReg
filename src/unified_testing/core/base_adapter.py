"""
Base adapter class for algorithm integration
"""

from abc import ABC, abstractmethod
import torch
import numpy as np

from src.common.utils.benchmark_preprocess import (
    normalize_point_cloud_pair,
    recover_raw_transform,
)


class BaseAdapter(ABC):
    """Base class for algorithm adapters"""

    def __init__(self, args):
        """
        Args:
            args: algorithm configuration parameters (namespace or dict)
        """
        self.args = args
        self.model = None
        self._source_normalization_transform = np.eye(4, dtype=np.float64)
        self._target_normalization_transform = np.eye(4, dtype=np.float64)

        # setup device
        if hasattr(args, "device"):
            device_str = (
                args.device
                if isinstance(args, dict)
                else getattr(args, "device", "cuda:0")
            )
        else:
            device_str = "cuda:0"

        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

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
        """Normalize a point-cloud pair according to the adapter's mode."""

        normalize_mode = getattr(self, "normalize_mode", "none")
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)

        source_norm, target_norm, info, source_transform, target_transform = (
            normalize_point_cloud_pair(source, target, normalize_mode)
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

    @abstractmethod
    def load_model(self, model_path):
        """
        Load model from checkpoint

        Args:
            model_path: model checkpoint file path
        """
        pass

    @abstractmethod
    def preprocess(self, source, target):
        """
        Preprocess point clouds for model input

        Args:
            source: [N, 3] numpy array (source point cloud)
            target: [N, 3] numpy array (target point cloud)

        Returns:
            source_tensor: preprocessed source tensor
            target_tensor: preprocessed target tensor
        """
        pass

    @abstractmethod
    def forward(self, source_tensor, target_tensor):
        """
        Forward inference through the model

        Args:
            source_tensor: preprocessed source tensor
            target_tensor: preprocessed target tensor

        Returns:
            output: model output (algorithm-specific)
        """
        pass

    @abstractmethod
    def extract_transformation(self, output):
        """
        Extract transformation from model output

        Args:
            output: output from forward()

        Returns:
            R: [3, 3] rotation matrix (numpy array)
            t: [3,] translation vector (numpy array)
        """
        pass

    def get_data_format(self):
        """
        Return expected data format

        Returns:
            format: 'BN3' (batch, num_points, 3) or 'B3N' (batch, 3, num_points)
        """
        return "BN3"  # default [B, N, 3]

    def get_algorithm_name(self):
        """
        Return algorithm name

        Returns:
            name: algorithm name string
        """
        return self.__class__.__name__.replace("Adapter", "")

    def set_eval_mode(self):
        """Set model to evaluation mode"""
        if self.model is not None:
            self.model.eval()

    def to_device(self, tensor):
        """
        Move tensor to device

        Args:
            tensor: torch.Tensor

        Returns:
            tensor: tensor on device
        """
        return tensor.to(self.device)

    def to_numpy(self, tensor):
        """
        Convert tensor to numpy array

        Args:
            tensor: torch.Tensor

        Returns:
            array: numpy array
        """
        return tensor.detach().cpu().numpy()

    def predict(self, source, target):
        """
        Full prediction pipeline: preprocess -> forward -> extract

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
        source_tensor, target_tensor = self.preprocess(source, target)

        # forward
        output = self.forward(source_tensor, target_tensor)

        # extract transformation
        R, t = self.extract_transformation(output)

        return R, t

    def __str__(self):
        return f"{self.get_algorithm_name()} (device: {self.device}, format: {self.get_data_format()})"
