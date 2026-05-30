#!/usr/bin/env python
# ruff: noqa: E402,E501,I001
"""
Unified testing script for point cloud registration algorithms

Usage:
    python unified_test.py --config configs/dcp_modelnet.yaml
    python unified_test.py --config configs/pointnetlk_c3vd_mamba3d.yaml
"""

import argparse
import os
import sys
import time
import yaml
import json
import numpy as np
import torch
from tqdm import tqdm

# add current directory and parent to path
UNIFIED_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, UNIFIED_TEST_DIR)
sys.path.insert(0, os.path.dirname(UNIFIED_TEST_DIR))

from unified_testing.core import DatasetLoader, PerturbationManager, ResultManager
from unified_testing.adapters import (
    POINTNETLK_C3VD_IMPORT_ERROR,
    PointNetLKC3VDAdapter,
    PointNetLKAdapter,
    PointNetLKRevisitedAdapter,
    BufferXAdapter,
    GeoTransformerAdapter,
    DCPAdapter,
    ICPAdapter,
)
from unified_testing.utils import (
    twist_to_se3,
    apply_transform,
    compute_all_errors,
    setup_logger,
    get_timestamp,
)


# Adapter registry
ADAPTERS = {
    "pointnetlk": PointNetLKAdapter,
    "pointnetlk_revisited": PointNetLKRevisitedAdapter,
    "bufferx": BufferXAdapter,
    "geotransformer": GeoTransformerAdapter,
    "dcp": DCPAdapter,
    "icp": ICPAdapter,
}

if PointNetLKC3VDAdapter is not None:
    ADAPTERS["pointnetlk_c3vd"] = PointNetLKC3VDAdapter


def load_config(config_file):
    """Load YAML configuration file"""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def dict_to_namespace(d):
    """Convert dict to namespace for compatibility"""
    from argparse import Namespace

    return Namespace(**d)


def run_unified_test(config):
    """
    Run unified testing pipeline

    Args:
        config: configuration dict from YAML file
    """
    # setup logger
    timestamp = get_timestamp()
    log_dir = os.path.join(config["output"]["dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"test_{timestamp}.log")
    logger = setup_logger("unified_test", log_file)

    logger.info("=" * 80)
    logger.info("UNIFIED TESTING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config}")

    # ============================================================
    # 1. Load Dataset
    # ============================================================
    logger.info("\n[1/5] Loading dataset...")
    dataset_config = config["dataset"]

    dataset = DatasetLoader(
        dataset_type=dataset_config["type"],
        data_root=dataset_config["data_root"],
        num_points=dataset_config.get("num_points", 1024),
        split=dataset_config.get("split", "test"),
        **dataset_config.get("kwargs", {}),
    )

    logger.info(f"✓ Loaded {dataset_config['type']} dataset: {len(dataset)} samples")

    # ============================================================
    # 2. Load Perturbations
    # ============================================================
    logger.info("\n[2/5] Loading perturbations...")
    pert_config = config["perturbation"]

    pert_manager = PerturbationManager(
        perturbation_path=pert_config["path"],
        random_seed=pert_config.get("random_seed", 42),
        angle_filter=pert_config.get("angle_filter", None),  # Support angle filtering
    )

    logger.info(f"✓ Loaded {pert_manager.get_num_files()} perturbation file(s)")

    # ============================================================
    # 3. Initialize Algorithm Adapter
    # ============================================================
    logger.info("\n[3/5] Initializing algorithm...")
    algo_config = config["algorithm"]
    algo_name = algo_config["name"]

    if algo_name == "pointnetlk_c3vd" and PointNetLKC3VDAdapter is None:
        raise ImportError(
            "pointnetlk_c3vd adapter is unavailable in the current environment. "
            f"Original import error: {POINTNETLK_C3VD_IMPORT_ERROR}"
        )

    if algo_name not in ADAPTERS:
        raise ValueError(
            f"Unknown algorithm: {algo_name}. Available: {list(ADAPTERS.keys())}"
        )

    # convert config dict to namespace
    algo_args = dict_to_namespace(algo_config.get("config", {}))
    algo_args.device = config.get("device", "cuda:0")

    # create adapter
    adapter = ADAPTERS[algo_name](algo_args)

    # load model
    model_path = algo_config.get("model_path", None)
    adapter.load_model(model_path)

    logger.info(f"✓ Initialized {adapter}")

    # ============================================================
    # 4. Run Testing
    # ============================================================
    logger.info("\n[4/5] Running tests...")

    # set random seed
    random_seed = pert_config.get("random_seed", 42)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # track all angle summaries for overall summary
    all_angle_summaries = []

    # Get max_perturbations from config (default: 1000)
    max_perturbations = config.get("max_perturbations", 1000)
    logger.info(f"Max perturbations per angle: {max_perturbations}")

    # iterate over perturbation files
    for perturbations, pert_metadata in pert_manager.iterate_perturbations(
        max_perturbations=max_perturbations
    ):
        angle = pert_metadata["angle"]
        num_perts = pert_metadata["num_perturbations"]

        logger.info(f"\nProcessing perturbations: {pert_metadata['filename']}")
        logger.info(f"  Angle: {angle}°, Num perturbations: {num_perts}")

        # create NEW result_manager for each angle
        result_manager = ResultManager(config["output"]["dir"])

        # random sampling: select one sample per perturbation
        for pert_idx in tqdm(range(num_perts), desc=f"Angle {angle}"):
            # randomly select a sample from dataset
            sample_idx = np.random.randint(0, len(dataset))
            data = dataset[sample_idx]

            source_original = data["source"]  # [N, 3]
            target = data["target"]  # [N, 3]
            metadata = data["metadata"]

            # get perturbation
            twist = perturbations[pert_idx]  # [6,] (rx, ry, rz, tx, ty, tz)
            g_pert = twist_to_se3(twist)  # [4, 4]

            # Check if adapter supports normalized-space perturbation (matching training)
            # This is the correct way to test models trained with normalization
            if (
                hasattr(adapter, "supports_normalized_perturbation")
                and adapter.supports_normalized_perturbation()
            ):
                # Method 1: Apply perturbation AFTER normalization (matches training for C3VD models)
                # 1. Preprocess without perturbation (includes normalization)
                source_norm, target_norm, preprocess_info = (
                    adapter.preprocess_for_perturbation(source_original, target)
                )

                # 2. Apply perturbation in normalized space
                source_perturbed = apply_transform(source_norm, g_pert)

                # 3. Run inference (no additional preprocessing)
                start_time = time.time()
                try:
                    R_pred, t_pred = adapter.predict_after_perturbation(
                        source_perturbed, target_norm, preprocess_info
                    )
                    inference_time = time.time() - start_time
                except Exception as e:
                    logger.error(f"Error during inference: {e}")
                    continue

                # Compute errors in normalized space
                R_gt = g_pert[:3, :3].T
                t_gt = -R_gt @ g_pert[:3, 3]
                errors = compute_all_errors(
                    source_perturbed, target_norm, R_pred, t_pred, R_gt, t_gt
                )
            else:
                # Method 2: Apply perturbation BEFORE normalization (original method)
                # This is used for models not trained with normalization or for backward compatibility
                source_perturbed = apply_transform(source_original, g_pert)

                # run inference
                start_time = time.time()
                try:
                    R_pred, t_pred = adapter.predict(source_perturbed, target)
                    inference_time = time.time() - start_time
                except Exception as e:
                    logger.error(f"Error during inference: {e}")
                    continue

                # compute ground truth (inverse of perturbation)
                R_gt = g_pert[:3, :3].T
                t_gt = -R_gt @ g_pert[:3, 3]

                # compute errors
                errors = compute_all_errors(
                    source_perturbed, target, R_pred, t_pred, R_gt, t_gt
                )

            # prepare result entry
            result_entry = {
                "pert_idx": pert_idx,
                "sample_idx": sample_idx,
                "dataset": dataset_config["type"],
                "algorithm": adapter.get_algorithm_name(),
                "angle": angle if angle is not None else -1,
                "source_file": metadata.get("source_file", "N/A"),
                "target_file": metadata.get("target_file", "N/A"),
                "perturbed_cloud": "source",
                # perturbation
                "pert_rx": twist[0],
                "pert_ry": twist[1],
                "pert_rz": twist[2],
                "pert_tx": twist[3],
                "pert_ty": twist[4],
                "pert_tz": twist[5],
                # prediction
                "pred_r11": R_pred[0, 0],
                "pred_r12": R_pred[0, 1],
                "pred_r13": R_pred[0, 2],
                "pred_r21": R_pred[1, 0],
                "pred_r22": R_pred[1, 1],
                "pred_r23": R_pred[1, 2],
                "pred_r31": R_pred[2, 0],
                "pred_r32": R_pred[2, 1],
                "pred_r33": R_pred[2, 2],
                "pred_tx": t_pred[0],
                "pred_ty": t_pred[1],
                "pred_tz": t_pred[2],
                # errors
                "rotation_error_deg": errors["rotation_error_deg"],
                "translation_error_m": errors["translation_error"],
                "se3_distance": errors["se3_distance"],
                "point_mse": errors["point_mse"],
                "inference_time_s": inference_time,
            }

            result_manager.add_result(result_entry)

        # ============================================================
        # Save results immediately after each angle completes
        # ============================================================
        logger.info(f"\nSaving results for angle {angle}°...")

        if result_manager.get_num_results() == 0:
            logger.warning(
                f"No successful results for angle {angle}°. Skipping CSV/summary generation for this angle."
            )
            continue

        # save CSV for this angle
        csv_file = os.path.join(
            config["output"]["dir"], f"results_angle_{angle:02d}.csv"
        )
        result_manager.save_results(csv_file)

        # save summary for this angle
        if config["output"].get("save_summary", True):
            summary_file = os.path.join(
                config["output"]["dir"], f"summary_angle_{angle:02d}.json"
            )
            angle_summary = result_manager.generate_summary(summary_file)
            # add perturbation angle to summary
            angle_summary["perturbation_angle_deg"] = angle
            all_angle_summaries.append(angle_summary)

    # ============================================================
    # 5. Generate Overall Summary
    # ============================================================
    logger.info("\n[5/5] Generating overall summary...")

    if config["output"].get("save_summary", True) and len(all_angle_summaries) > 0:
        # compute overall statistics across all angles
        overall_summary = {
            "algorithm": all_angle_summaries[0]["algorithm"],
            "dataset": all_angle_summaries[0]["dataset"],
            "num_angles": len(all_angle_summaries),
            "total_samples": sum(s["num_samples"] for s in all_angle_summaries),
            "per_angle_summaries": all_angle_summaries,
            "overall_statistics": {
                "rotation_error": {
                    "mean_across_angles": float(
                        np.mean(
                            [s["rotation_error"]["mean"] for s in all_angle_summaries]
                        )
                    ),
                    "std_across_angles": float(
                        np.std(
                            [s["rotation_error"]["mean"] for s in all_angle_summaries]
                        )
                    ),
                    "min_mean": float(
                        np.min(
                            [s["rotation_error"]["mean"] for s in all_angle_summaries]
                        )
                    ),
                    "max_mean": float(
                        np.max(
                            [s["rotation_error"]["mean"] for s in all_angle_summaries]
                        )
                    ),
                },
                "translation_error": {
                    "mean_across_angles": float(
                        np.mean(
                            [
                                s["translation_error"]["mean"]
                                for s in all_angle_summaries
                            ]
                        )
                    ),
                    "std_across_angles": float(
                        np.std(
                            [
                                s["translation_error"]["mean"]
                                for s in all_angle_summaries
                            ]
                        )
                    ),
                    "min_mean": float(
                        np.min(
                            [
                                s["translation_error"]["mean"]
                                for s in all_angle_summaries
                            ]
                        )
                    ),
                    "max_mean": float(
                        np.max(
                            [
                                s["translation_error"]["mean"]
                                for s in all_angle_summaries
                            ]
                        )
                    ),
                },
                "inference_time": {
                    "total_time": float(
                        sum(s["inference_time"]["total"] for s in all_angle_summaries)
                    ),
                    "mean_per_sample": float(
                        np.mean(
                            [s["inference_time"]["mean"] for s in all_angle_summaries]
                        )
                    ),
                },
            },
        }

        # save overall summary
        overall_summary_file = os.path.join(
            config["output"]["dir"], f"summary_overall_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(overall_summary_file), exist_ok=True)
        with open(overall_summary_file, "w") as f:
            json.dump(overall_summary, f, indent=2)
        print(f"✓ Saved overall summary to: {overall_summary_file}")

        # print overall statistics
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY ACROSS ALL ANGLES")
        print("=" * 60)
        print(f"Number of angles: {overall_summary['num_angles']}")
        print(f"Total samples: {overall_summary['total_samples']}")
        print("\nRotation Error (mean across angles):")
        print(
            f"  Mean ± Std: {overall_summary['overall_statistics']['rotation_error']['mean_across_angles']:.3f} ± {overall_summary['overall_statistics']['rotation_error']['std_across_angles']:.3f}"
        )
        print(
            f"  Min / Max: {overall_summary['overall_statistics']['rotation_error']['min_mean']:.3f} / {overall_summary['overall_statistics']['rotation_error']['max_mean']:.3f}"
        )
        print("\nTranslation Error (mean across angles):")
        print(
            f"  Mean ± Std: {overall_summary['overall_statistics']['translation_error']['mean_across_angles']:.4f} ± {overall_summary['overall_statistics']['translation_error']['std_across_angles']:.4f}"
        )
        print(
            f"  Min / Max: {overall_summary['overall_statistics']['translation_error']['min_mean']:.4f} / {overall_summary['overall_statistics']['translation_error']['max_mean']:.4f}"
        )
        print(
            "\nTotal Inference Time: {:.2f}s".format(
                overall_summary["overall_statistics"]["inference_time"]["total_time"]
            )
        )
        print("=" * 60 + "\n")

    logger.info("\n" + "=" * 80)
    logger.info("TESTING COMPLETED")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Unified testing pipeline for point cloud registration"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    # check config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # load config
    config = load_config(args.config)

    # run testing
    run_unified_test(config)


if __name__ == "__main__":
    main()
