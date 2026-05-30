"""
Perturbation manager for handling perturbation files
"""

import os
import glob
import numpy as np


class PerturbationManager:
    """Manager for perturbation files"""

    def __init__(self, perturbation_path, random_seed=42, angle_filter=None):
        """
        Args:
            perturbation_path: perturbation file or directory path
            random_seed: random seed for reproducibility
            angle_filter: list of angles to filter (e.g., [40, 50, 60]), None means all
        """
        self.random_seed = random_seed
        self.angle_filter = angle_filter
        np.random.seed(random_seed)

        # scan perturbation files
        self.perturbation_files = self._scan_perturbation_files(perturbation_path)

        if not self.perturbation_files:
            raise ValueError(f"No perturbation files found at: {perturbation_path}")

        print(f"Found {len(self.perturbation_files)} perturbation file(s)")
        if angle_filter:
            print(f"Filtering for angles: {angle_filter}")

    def _scan_perturbation_files(self, path):
        """Scan for perturbation files"""
        files = []

        if os.path.isfile(path):
            # single file
            files.append(path)
        elif os.path.isdir(path):
            # directory: scan all .csv files
            pattern = os.path.join(path, "*.csv")
            files = sorted(glob.glob(pattern))
        else:
            raise ValueError(f"Invalid path: {path}")

        return files

    def load_perturbations(self, file_path, max_perturbations=None):
        """
        Load a single perturbation file

        Args:
            file_path: perturbation file path
            max_perturbations: maximum number of perturbations to load (None = all)

        Returns:
            perturbations: [M, 6] numpy array
            metadata: dict with file info
        """
        perturbations = np.loadtxt(file_path, delimiter=",")

        # ensure correct shape
        if perturbations.ndim == 1:
            perturbations = perturbations.reshape(1, -1)

        # limit number of perturbations if specified
        if max_perturbations is not None and len(perturbations) > max_perturbations:
            perturbations = perturbations[:max_perturbations]

        # extract angle information (if any)
        filename = os.path.basename(file_path)
        angle = self._extract_angle_from_filename(filename)

        metadata = {
            "file_path": file_path,
            "filename": filename,
            "angle": angle,
            "num_perturbations": len(perturbations),
        }

        return perturbations, metadata

    def _extract_angle_from_filename(self, filename):
        """
        Extract angle from filename

        Example: pert_030.csv -> 30
        """
        if filename.startswith("pert_") and "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 2:
                angle_str = parts[1].split(".")[0]
                try:
                    return int(angle_str)
                except ValueError:
                    pass

        return None

    def iterate_perturbations(self, max_perturbations=None):
        """
        Iterate over all perturbation files

        Args:
            max_perturbations: maximum number of perturbations per file (None = all)

        Yields:
            perturbations, metadata
        """
        for file_path in self.perturbation_files:
            perturbations, metadata = self.load_perturbations(
                file_path, max_perturbations=max_perturbations
            )

            # Apply angle filter if specified
            if self.angle_filter is not None:
                angle = metadata["angle"]
                if angle is None or angle not in self.angle_filter:
                    continue  # Skip this file

            yield perturbations, metadata

    def get_num_files(self):
        """Get number of perturbation files"""
        return len(self.perturbation_files)

    def get_file_paths(self):
        """Get list of all perturbation file paths"""
        return self.perturbation_files.copy()
