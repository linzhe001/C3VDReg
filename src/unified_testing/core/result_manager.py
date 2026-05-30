"""
Result manager for saving and managing test results
"""

import os
import json
import pandas as pd
import numpy as np


class ResultManager:
    """Manager for test results"""

    def __init__(self, output_dir):
        """
        Args:
            output_dir: output directory
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.results = []

    def add_result(self, result_dict):
        """
        Add a result entry

        Args:
            result_dict: dict with result data (must contain all required fields)
        """
        self.results.append(result_dict)

    def save_results(self, output_file):
        """
        Save results to CSV file

        Args:
            output_file: output CSV file path

        Returns:
            df: pandas DataFrame with results
        """
        # create DataFrame
        df = pd.DataFrame(self.results)

        # ensure correct column order (31 columns as specified)
        columns = [
            "pert_idx",
            "sample_idx",
            "dataset",
            "algorithm",
            "angle",
            "source_file",
            "target_file",
            "perturbed_cloud",
            "pert_rx",
            "pert_ry",
            "pert_rz",
            "pert_tx",
            "pert_ty",
            "pert_tz",
            "pred_r11",
            "pred_r12",
            "pred_r13",
            "pred_r21",
            "pred_r22",
            "pred_r23",
            "pred_r31",
            "pred_r32",
            "pred_r33",
            "pred_tx",
            "pred_ty",
            "pred_tz",
            "rotation_error_deg",
            "translation_error_m",
            "se3_distance",
            "point_mse",
            "inference_time_s",
        ]

        # check if all columns are present
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in results: {missing_cols}")

        df = df[columns]

        # save CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"✓ Saved results to: {output_file}")

        return df

    def generate_summary(self, summary_file):
        """
        Generate summary JSON file

        Args:
            summary_file: output JSON file path

        Returns:
            summary: dict with statistics
        """
        df = pd.DataFrame(self.results)

        if len(df) == 0:
            print("Warning: No results to summarize")
            return {}

        # compute statistics
        summary = {
            "num_samples": len(df),
            "algorithm": df["algorithm"].iloc[0] if len(df) > 0 else "unknown",
            "dataset": df["dataset"].iloc[0] if len(df) > 0 else "unknown",
            "rotation_error": {
                "mean": float(df["rotation_error_deg"].mean()),
                "std": float(df["rotation_error_deg"].std()),
                "median": float(df["rotation_error_deg"].median()),
                "min": float(df["rotation_error_deg"].min()),
                "max": float(df["rotation_error_deg"].max()),
            },
            "translation_error": {
                "mean": float(df["translation_error_m"].mean()),
                "std": float(df["translation_error_m"].std()),
                "median": float(df["translation_error_m"].median()),
                "min": float(df["translation_error_m"].min()),
                "max": float(df["translation_error_m"].max()),
            },
            "se3_distance": {
                "mean": float(df["se3_distance"].mean()),
                "std": float(df["se3_distance"].std()),
                "median": float(df["se3_distance"].median()),
            },
            "point_mse": {
                "mean": float(df["point_mse"].mean()),
                "std": float(df["point_mse"].std()),
                "median": float(df["point_mse"].median()),
            },
            "inference_time": {
                "mean": float(df["inference_time_s"].mean()),
                "std": float(df["inference_time_s"].std()),
                "total": float(df["inference_time_s"].sum()),
            },
        }

        # save JSON
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Saved summary to: {summary_file}")

        return summary

    def clear(self):
        """Clear all results"""
        self.results = []

    def get_num_results(self):
        """Get number of results"""
        return len(self.results)

    def print_summary_stats(self):
        """Print summary statistics to console"""
        if len(self.results) == 0:
            print("No results available")
            return

        df = pd.DataFrame(self.results)

        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Number of samples: {len(df)}")
        print(f"Algorithm: {df['algorithm'].iloc[0]}")
        print(f"Dataset: {df['dataset'].iloc[0]}")
        print("\nRotation Error (degrees):")
        print(
            f"  Mean ± Std: {df['rotation_error_deg'].mean():.3f} ± {df['rotation_error_deg'].std():.3f}"
        )
        print(f"  Median: {df['rotation_error_deg'].median():.3f}")
        print(
            f"  Min / Max: {df['rotation_error_deg'].min():.3f} / {df['rotation_error_deg'].max():.3f}"
        )
        print("\nTranslation Error (m):")
        print(
            f"  Mean ± Std: {df['translation_error_m'].mean():.4f} ± {df['translation_error_m'].std():.4f}"
        )
        print(f"  Median: {df['translation_error_m'].median():.4f}")
        print(
            f"  Min / Max: {df['translation_error_m'].min():.4f} / {df['translation_error_m'].max():.4f}"
        )
        print("\nInference Time (s):")
        print(
            f"  Mean ± Std: {df['inference_time_s'].mean():.4f} ± {df['inference_time_s'].std():.4f}"
        )
        print(f"  Total: {df['inference_time_s'].sum():.2f}")
        print("=" * 60 + "\n")
