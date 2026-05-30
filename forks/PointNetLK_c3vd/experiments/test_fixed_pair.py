#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定点云对配准测试脚本
Fixed Point Cloud Pair Registration Test Script

该脚本用于测试固定的点云对在给定扰动下的配准效果
This script is used to test the registration effect of fixed point cloud pairs under given perturbations
"""

import argparse
import torch
import numpy as np
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def options():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(
        description="固定点云对配准测试 - Fixed Point Cloud Pair Registration Test"
    )

    # 必需参数
    parser.add_argument(
        "--source-cloud",
        required=True,
        type=str,
        help="源点云文件路径 Source point cloud file path",
    )
    parser.add_argument(
        "--target-cloud",
        required=True,
        type=str,
        help="目标点云文件路径 Target point cloud file path",
    )
    parser.add_argument(
        "--model-path", required=True, type=str, help="模型文件路径 Model file path"
    )
    parser.add_argument(
        "--perturbation",
        required=True,
        type=str,
        help="扰动值，逗号分隔(rx,ry,rz,tx,ty,tz) Perturbation values, comma-separated",
    )

    # 输出设置
    parser.add_argument(
        "--output-csv",
        default=None,
        type=str,
        help="输出CSV文件路径 Output CSV file path",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/fixed_pair_test",
        type=str,
        help="输出目录 Output directory",
    )

    # 模型配置
    parser.add_argument(
        "--model-type",
        default="mamba3d",
        choices=[
            "pointnet",
            "attention",
            "mamba3d",
            "mamba3d_v2",
            "fast_attention",
            "cformer",
        ],
        help="模型类型 Model type",
    )
    parser.add_argument(
        "--dim-k", default=1024, type=int, help="特征维度 Feature dimension"
    )
    parser.add_argument(
        "--num-mamba-blocks",
        default=1,
        type=int,
        help="Mamba块数量 Number of Mamba blocks",
    )
    parser.add_argument(
        "--d-state", default=8, type=int, help="状态空间维度 State space dimension"
    )
    parser.add_argument(
        "--expand", default=2, type=float, help="扩展因子 Expansion factor"
    )
    parser.add_argument(
        "--symfn",
        default="max",
        choices=["max", "avg"],
        help="对称函数 Symmetric function",
    )

    # 算法参数
    parser.add_argument(
        "--max-iter", default=20, type=int, help="LK最大迭代次数 Maximum LK iterations"
    )
    parser.add_argument(
        "--delta", default=1.0e-4, type=float, help="LK步长 LK step size"
    )
    parser.add_argument(
        "--num-points",
        default=1024,
        type=int,
        help="点云点数 Number of points in point cloud",
    )

    # 设备设置
    parser.add_argument(
        "--device", default="cuda:0", type=str, help="计算设备 Computing device"
    )

    # 体素化参数
    parser.add_argument(
        "--use-voxelization",
        action="store_true",
        default=False,
        help="启用体素化 Enable voxelization",
    )
    parser.add_argument(
        "--voxel-size", default=4, type=float, help="体素大小 Voxel size"
    )
    parser.add_argument(
        "--voxel-grid-size", default=32, type=int, help="体素网格尺寸 Voxel grid size"
    )
    parser.add_argument(
        "--max-voxel-points",
        default=100,
        type=int,
        help="每个体素最大点数 Maximum points per voxel",
    )
    parser.add_argument(
        "--max-voxels",
        default=20000,
        type=int,
        help="最大体素数量 Maximum number of voxels",
    )
    parser.add_argument(
        "--min-voxel-points-ratio",
        default=0.1,
        type=float,
        help="最小体素点数比例 Minimum voxel points ratio",
    )

    # 调试选项
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="详细输出 Verbose output"
    )
    parser.add_argument(
        "--save-clouds",
        action="store_true",
        default=False,
        help="保存处理后的点云 Save processed point clouds",
    )

    return parser.parse_args()


def load_model(args):
    """加载模型"""
    from ptlk import pointlk
    from ptlk.data.datasets import VoxelizationConfig

    print(f"正在加载模型: {args.model_path}")

    # 创建体素化配置
    voxel_config = VoxelizationConfig(
        voxel_size=args.voxel_size,
        voxel_grid_size=args.voxel_grid_size,
        max_voxel_points=args.max_voxel_points,
        max_voxels=args.max_voxels,
        min_voxel_points_ratio=args.min_voxel_points_ratio,
    )

    # 根据模型类型创建模型
    if args.model_type == "pointnet":
        from ptlk.PointNet_files.pointnet_original import PointNet as FeatureModel

        ptnet = FeatureModel(args.dim_k)
    elif args.model_type == "mamba3d":
        from ptlk.data.mamba3d import Mamba3D

        ptnet = Mamba3D(
            dim_k=args.dim_k,
            num_blocks=args.num_mamba_blocks,
            d_state=args.d_state,
            expand=args.expand,
            symfn=args.symfn,
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")

    # 创建PointNetLK模型
    model = pointlk.PointNetLK(
        feature_model=ptnet,
        delta=args.delta,
        xtol=1.0e-7,
        p0_zero_mean=True,
        p1_zero_mean=True,
        pooling="max",
        use_voxelization=args.use_voxelization,
        voxel_config=voxel_config,
    )

    # 加载预训练权重
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✅ 成功加载模型权重")
    else:
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

    # 移动到指定设备
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    return model, device


def load_point_cloud(file_path, num_points=1024):
    """加载点云文件"""
    import plyfile

    print(f"正在加载点云: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"点云文件不存在: {file_path}")

    # 读取PLY文件
    plydata = plyfile.PlyData.read(file_path)
    vertices = plydata["vertex"]

    # 提取XYZ坐标
    points = np.column_stack(
        [
            vertices["x"].astype(np.float32),
            vertices["y"].astype(np.float32),
            vertices["z"].astype(np.float32),
        ]
    )

    print(f"原始点云形状: {points.shape}")

    # 重采样到指定点数
    if len(points) > num_points:
        # 随机采样
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    elif len(points) < num_points:
        # 重复采样
        indices = np.random.choice(len(points), num_points, replace=True)
        points = points[indices]

    print(f"重采样后点云形状: {points.shape}")

    return points


def create_perturbation_matrix(perturbation_values):
    """创建扰动变换矩阵"""
    import ptlk.se3 as se3

    # 解析扰动值 (rx,ry,rz,tx,ty,tz)
    rx, ry, rz, tx, ty, tz = perturbation_values

    # 创建旋转和平移
    rotation = se3.euler_to_so3(rx, ry, rz)  # 欧拉角转旋转矩阵
    translation = np.array([tx, ty, tz])

    # 构建4x4变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = translation

    return transform_matrix


def apply_perturbation(points, perturbation_matrix):
    """对点云应用扰动"""
    # 转换为齐次坐标
    points_homo = np.column_stack([points, np.ones(len(points))])

    # 应用变换
    transformed_points = (perturbation_matrix @ points_homo.T).T

    # 返回3D坐标
    return transformed_points[:, :3]


def compute_registration_error(gt_transform, pred_transform):
    """计算配准误差"""
    import ptlk.se3 as se3

    # 计算相对变换
    relative_transform = np.linalg.inv(gt_transform) @ pred_transform

    # 分解为旋转和平移
    rotation_part = relative_transform[:3, :3]
    translation_part = relative_transform[:3, 3]

    # 计算旋转误差 (角度)
    rotation_error = np.arccos(np.clip((np.trace(rotation_part) - 1) / 2, -1, 1))
    rotation_error_deg = np.degrees(rotation_error)

    # 计算平移误差 (欧几里得距离)
    translation_error = np.linalg.norm(translation_part)

    return rotation_error_deg, translation_error


def main():
    """主函数"""
    args = options()

    print("========== 固定点云对配准测试 ==========")
    print(f"源点云: {args.source_cloud}")
    print(f"目标点云: {args.target_cloud}")
    print(f"模型: {args.model_path}")
    print(f"扰动: {args.perturbation}")
    print(f"设备: {args.device}")
    print("")

    # 解析扰动值
    try:
        perturbation_values = [float(x.strip()) for x in args.perturbation.split(",")]
        if len(perturbation_values) != 6:
            raise ValueError(
                f"扰动值必须是6个数字，当前提供了{len(perturbation_values)}个"
            )
        print(f"解析的扰动值: {perturbation_values}")
    except Exception as e:
        print(f"❌ 扰动值解析错误: {e}")
        return

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # 加载模型
        print("\n1. 加载模型...")
        model, device = load_model(args)

        # 加载点云
        print("\n2. 加载点云...")
        source_points = load_point_cloud(args.source_cloud, args.num_points)
        target_points = load_point_cloud(args.target_cloud, args.num_points)

        # 创建扰动矩阵
        print("\n3. 创建扰动...")
        perturbation_matrix = create_perturbation_matrix(perturbation_values)
        print(f"扰动矩阵:\n{perturbation_matrix}")

        # 对源点云应用扰动
        source_points_perturbed = apply_perturbation(source_points, perturbation_matrix)

        # 准备数据
        template = (
            torch.from_numpy(target_points.T).float().unsqueeze(0).to(device)
        )  # [1, 3, N]
        source = (
            torch.from_numpy(source_points_perturbed.T).float().unsqueeze(0).to(device)
        )  # [1, 3, N]

        print(f"模板点云形状: {template.shape}")
        print(f"源点云形状: {source.shape}")

        # 执行配准
        print("\n4. 执行配准...")
        start_time = time.time()

        with torch.no_grad():
            result = model(template, source, maxiter=args.max_iter)

        end_time = time.time()
        registration_time = end_time - start_time
        print(f"配准耗时: {registration_time:.4f} 秒")

        # 提取结果
        if isinstance(result, tuple):
            predicted_transform = result[0]
        else:
            predicted_transform = result

        # 转换为numpy数组
        predicted_transform_np = predicted_transform.cpu().numpy().squeeze()

        print(f"预测变换矩阵形状: {predicted_transform_np.shape}")
        print(f"预测变换矩阵:\n{predicted_transform_np}")

        # 计算配准误差
        print("\n5. 计算配准误差...")
        gt_transform = np.linalg.inv(perturbation_matrix)  # 真实变换是扰动的逆
        rotation_error, translation_error = compute_registration_error(
            gt_transform, predicted_transform_np
        )

        print(f"旋转误差: {rotation_error:.6f} 度")
        print(f"平移误差: {translation_error:.6f} mm")

        # 保存结果
        print("\n6. 保存结果...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 保存到CSV
        if args.output_csv:
            output_csv = args.output_csv
        else:
            output_csv = os.path.join(
                args.output_dir, f"fixed_pair_result_{timestamp}.csv"
            )

        # 写入CSV文件
        with open(output_csv, "w") as f:
            f.write(
                "source_cloud,target_cloud,perturbation,rotation_error_deg,translation_error_mm,registration_time_sec\n"
            )
            f.write(
                f"{os.path.basename(args.source_cloud)},{os.path.basename(args.target_cloud)},{args.perturbation},{rotation_error:.6f},{translation_error:.6f},{registration_time:.4f}\n"
            )

        print(f"✅ 结果已保存到: {output_csv}")

        # 保存详细信息
        detail_file = os.path.join(
            args.output_dir, f"fixed_pair_detail_{timestamp}.txt"
        )
        with open(detail_file, "w") as f:
            f.write("=== 固定点云对配准测试详细结果 ===\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"源点云: {args.source_cloud}\n")
            f.write(f"目标点云: {args.target_cloud}\n")
            f.write(f"模型: {args.model_path}\n")
            f.write(f"模型类型: {args.model_type}\n")
            f.write(f"设备: {args.device}\n")
            f.write(f"点云点数: {args.num_points}\n")
            f.write(f"最大迭代: {args.max_iter}\n")
            f.write(f"扰动值: {perturbation_values}\n")
            f.write(f"扰动矩阵:\n{perturbation_matrix}\n")
            f.write(f"预测变换矩阵:\n{predicted_transform_np}\n")
            f.write(f"旋转误差: {rotation_error:.6f} 度\n")
            f.write(f"平移误差: {translation_error:.6f} mm\n")
            f.write(f"配准耗时: {registration_time:.4f} 秒\n")

        print(f"✅ 详细结果已保存到: {detail_file}")

        # 可选：保存点云
        if args.save_clouds:
            print("\n7. 保存点云...")
            cloud_dir = os.path.join(args.output_dir, f"clouds_{timestamp}")
            os.makedirs(cloud_dir, exist_ok=True)

            np.savetxt(os.path.join(cloud_dir, "source_original.txt"), source_points)
            np.savetxt(
                os.path.join(cloud_dir, "source_perturbed.txt"), source_points_perturbed
            )
            np.savetxt(os.path.join(cloud_dir, "target.txt"), target_points)

            print(f"✅ 点云已保存到: {cloud_dir}")

        print("\n🎉 测试完成!")
        print(f"📊 旋转误差: {rotation_error:.6f} 度")
        print(f"📊 平移误差: {translation_error:.6f} mm")
        print(f"⏱️  配准耗时: {registration_time:.4f} 秒")

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
