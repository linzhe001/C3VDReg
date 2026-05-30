import torch
import numpy as np

"""
PointMamba Adapter with Full Hilbert Serialization
完整的 Hilbert 序列化实现

文件: pointmamba_adapter_hilbert.py
功能: Level 2 中度集成 + 完整 Hilbert 序列化
"""

# ========== Hilbert 曲线编码实现 ==========


def _hilbert_encode(locs, num_dims=3, num_bits=16):
    """
    Hilbert 曲线编码
    将 3D 坐标编码为 Hilbert 曲线上的 1D 索引

    Args:
        locs: [N, 3] 整数坐标 (grid_coord)
        num_dims: 空间维度 (默认 3)
        num_bits: 每个维度的位数 (默认 16)

    Returns:
        hilbert_indices: [N] Hilbert 曲线索引

    算法来源: Skilling (2004) "Programming the Hilbert curve"
    """
    # 确保输入是整数
    locs = locs.long()

    # 检查位数限制
    if num_dims * num_bits > 63:
        raise ValueError(
            f"num_dims={num_dims} and num_bits={num_bits} for {num_dims * num_bits} bits total, "
            f"which can't be encoded into a int64."
        )

    # 初始化 Hilbert 索引
    device = locs.device
    num_locs = locs.shape[0]
    hilbert = torch.zeros(num_locs, dtype=torch.int64, device=device)

    # 位操作掩码
    one = torch.ones(1, dtype=torch.int64, device=device)

    # Gray code 转换
    for bit in range(num_bits - 1, -1, -1):
        # 提取当前位
        mask = one << bit
        bits = ((locs & mask) >> bit).long()  # [N, 3]

        # 计算 Gray code
        gray = bits[:, 0]
        for dim in range(1, num_dims):
            gray = gray ^ bits[:, dim]

        # 旋转和反射变换
        for dim in range(num_dims - 1, 0, -1):
            bits[:, dim] = bits[:, dim] ^ bits[:, dim - 1]

        # 计算方向
        t = bits[:, 0]
        for i in range(1, num_dims):
            t = t ^ bits[:, i]

        # 更新 Hilbert 索引
        for dim in range(num_dims):
            hilbert = (hilbert << 1) | bits[:, dim]

    return hilbert


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    """
    简化的 Hilbert 编码接口

    Args:
        grid_coord: [N, 3] 网格坐标
        depth: 位深度

    Returns:
        hilbert_code: [N] Hilbert 编码
    """
    return _hilbert_encode(grid_coord, num_dims=3, num_bits=depth)


def encode(grid_coord, batch=None, depth=16, order="hilbert"):
    """
    编码空间坐标为序列化代码

    Args:
        grid_coord: [N, 3] 网格坐标
        batch: [N] batch 索引 (可选)
        depth: 编码深度
        order: 'hilbert' 或 'hilbert-trans'

    Returns:
        code: [N] 序列化代码
    """
    if order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        # 转置坐标 (交换 x 和 y)
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError(f"Unknown order: {order}")

    # 如果有 batch，将 batch ID 编码到高位
    if batch is not None:
        batch = batch.long()
        code = (batch << (depth * 3)) | code

    return code


def serialization_func(
    center, group_input_tokens=None, pos=None, order="hilbert", grid_size=0.02
):
    """
    Hilbert 序列化函数

    Args:
        center: [B, G, 3] 点云中心坐标
        group_input_tokens: [B, G, D] 组特征 (可选)
        pos: [B, G, D] 位置编码 (可选)
        order: 序列化顺序 ('hilbert' 或 'hilbert-trans')
        grid_size: 网格大小 (默认 0.02)

    Returns:
        center_sorted: [B, G, 3] 排序后的中心
        order_indices: [B, G] 排序索引
        inverse_indices: [B, G] 逆序索引
        tokens_sorted: [B, G, D] 排序后的特征
        pos_sorted: [B, G, D] 排序后的位置编码
    """
    B, G, _ = center.shape
    device = center.device

    # 1. 空间量化：将连续坐标转换为网格坐标
    # 找到最小坐标作为原点
    center_min = center.reshape(B * G, 3).min(dim=0)[0]  # [3]

    # 量化到网格
    grid_coord = ((center - center_min.view(1, 1, 3)) / grid_size).long()  # [B, G, 3]
    grid_coord = grid_coord.clamp(min=0, max=(1 << 16) - 1)  # 限制到 16 位范围

    # 2. 为每个 batch 编码 Hilbert 曲线
    center_sorted_list = []
    order_indices_list = []
    inverse_indices_list = []
    tokens_sorted_list = [] if group_input_tokens is not None else None
    pos_sorted_list = [] if pos is not None else None

    for b in range(B):
        # 计算 Hilbert 编码
        hilbert_code = encode(
            grid_coord[b],  # [G, 3]
            batch=None,
            depth=16,
            order=order,
        )  # [G]

        # 根据 Hilbert 编码排序
        order_idx = torch.argsort(hilbert_code)  # [G]

        # 计算逆序索引
        inverse_idx = torch.zeros_like(order_idx)
        inverse_idx[order_idx] = torch.arange(G, device=device)

        # 应用排序
        center_sorted_list.append(center[b][order_idx])  # [G, 3]
        order_indices_list.append(order_idx)
        inverse_indices_list.append(inverse_idx)

        if group_input_tokens is not None:
            tokens_sorted_list.append(group_input_tokens[b][order_idx])  # [G, D]

        if pos is not None:
            pos_sorted_list.append(pos[b][order_idx])  # [G, D]

    # 3. 堆叠结果
    center_sorted = torch.stack(center_sorted_list, dim=0)  # [B, G, 3]
    order_indices = torch.stack(order_indices_list, dim=0)  # [B, G]
    inverse_indices = torch.stack(inverse_indices_list, dim=0)  # [B, G]

    tokens_sorted = (
        torch.stack(tokens_sorted_list, dim=0)
        if tokens_sorted_list is not None
        else None
    )
    pos_sorted = (
        torch.stack(pos_sorted_list, dim=0) if pos_sorted_list is not None else None
    )

    return center_sorted, order_indices, inverse_indices, tokens_sorted, pos_sorted
