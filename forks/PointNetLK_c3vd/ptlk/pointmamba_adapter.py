import math
from functools import partial

import torch
import torch.nn as nn

# ========== 依赖检查 ==========
try:
    from mamba_ssm.modules.mamba_simple import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba_ssm not found. Please install: pip install mamba-ssm")
    MAMBA_AVAILABLE = False

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

try:
    from knn_cuda import KNN

    KNN_CUDA_AVAILABLE = True
except Exception as exc:
    print(f"Warning: knn_cuda unavailable ({exc}). Using PyTorch fallback (slower)")
    KNN_CUDA_AVAILABLE = False


# ========== 核心组件（与之前相同）==========


class Encoder(nn.Module):
    """局部几何特征编码器"""

    def __init__(self, encoder_channel=384):
        super().__init__()
        self.encoder_channel = encoder_channel

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        Args:
            point_groups: [B, G, N, 3]
        Returns:
            feature_global: [B, G, C]
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)

        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]

        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    """FPS + KNN 分组"""

    def __init__(self, num_group=64, group_size=32):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

        if KNN_CUDA_AVAILABLE:
            self.knn = KNN(k=self.group_size, transpose_mode=True)
        else:
            print("Warning: Using fallback KNN (slow)")

    def forward(self, xyz):
        """
        Args:
            xyz: [B, N, 3]
        Returns:
            neighborhood: [B, G, M, 3]
            center: [B, G, 3]
        """
        batch_size, num_points, _ = xyz.shape

        # FPS 采样
        center = self._fps(xyz, self.num_group)

        # KNN 查询
        if KNN_CUDA_AVAILABLE:
            _, idx = self.knn(xyz, center)
        else:
            idx = self._knn_pytorch(xyz, center, self.group_size)

        # 索引邻域
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)

        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()

        # 中心化
        neighborhood = neighborhood - center.unsqueeze(2)

        return neighborhood, center

    def _fps(self, xyz, npoint):
        """Farthest Point Sampling"""
        device = xyz.device
        B, N, C = xyz.shape

        centroids = torch.zeros(B, npoint, C).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)

        for i in range(npoint):
            centroids[:, i, :] = xyz[batch_indices, farthest, :]
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]

        return centroids

    def _knn_pytorch(self, xyz, query, k):
        """PyTorch KNN fallback"""
        dist = torch.cdist(query, xyz)
        _, idx = torch.topk(dist, k, dim=2, largest=False, sorted=True)
        return idx


def _default_sym_fn(x):
    return torch.max(x, dim=1)[0]


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    """Mamba 权重初始化"""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    drop_path=0.0,
    device=None,
    dtype=None,
):
    """创建单个 Mamba 块"""
    if not MAMBA_AVAILABLE:
        raise ImportError("mamba_ssm is required for this module")

    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    # 导入 Block（需要确保路径正确）
    # 优先从本地导入，然后尝试 PointMamba，最后回退到 mamba_ssm
    try:
        from .block_scan import Block
    except ImportError:
        try:
            from PointMamba.models.block_scan import Block
        except ImportError:
            try:
                from block_scan import Block
            except ImportError:
                # 使用 mamba_ssm 的 Block 实现（不支持 drop_path）
                from mamba_ssm.modules.block import Block

                # 创建不带 drop_path 的 block
                mixer_cls = partial(
                    Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs
                )
                norm_cls = partial(
                    nn.LayerNorm if not rms_norm else RMSNorm,
                    eps=norm_epsilon,
                    **factory_kwargs,
                )
                block = Block(
                    d_model,
                    mixer_cls,
                    mlp_cls=None,  # Mamba 不需要 MLP
                    norm_cls=norm_cls,
                    fused_add_norm=fused_add_norm,
                    residual_in_fp32=residual_in_fp32,
                )
                block.layer_idx = layer_idx
                return block

    # 使用自定义 Block（支持 drop_path）
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )

    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block


class MixerModel(nn.Module):
    """Mamba 序列模型"""

    def __init__(
        self,
        d_model,
        n_layer,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_out=0.0,
        drop_path=0.0,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        self.drop_out = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Identity()

    def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids + pos

        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params)
            hidden_states = self.drop_out(hidden_states)

        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
        return hidden_states


def init_OrderScale(dim):
    """初始化 OrderScale 参数"""
    gamma = nn.Parameter(torch.ones(dim))
    beta = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(gamma, mean=1, std=0.02)
    nn.init.normal_(beta, std=0.02)
    return gamma, beta


def apply_OrderScale(x, gamma, beta):
    """应用 OrderScale 变换"""
    if x.shape[-1] == gamma.shape[0]:
        return x * gamma + beta
    elif x.shape[1] == gamma.shape[0]:
        return x * gamma.view(1, -1, 1, 1) + beta.view(1, -1, 1, 1)
    else:
        raise ValueError(
            f"Shape mismatch: x.shape={x.shape}, gamma.shape={gamma.shape}"
        )


# ========== 主类：PointMamba 特征提取器（带 Hilbert）==========


class PointMamba_features(nn.Module):
    """
    PointMamba 特征提取器（完整 Hilbert 序列化版本）

    Level 2: 中度集成 + 完整 Hilbert 序列化
    - 6 层 Mamba（vs 12 层完整版）
    - 完整 Hilbert 曲线序列化
    - 双向扫描（hilbert + hilbert-trans）
    """

    def __init__(
        self,
        dim_k=1024,
        sym_fn=None,
        scale=1,
        num_groups=64,
        group_size=32,
        trans_dim=384,
        depth=6,
        drop_path_rate=0.1,
        drop_out=0.0,
        rms_norm=False,
        grid_size=0.02,
    ):
        super().__init__()

        self.dim_k = int(dim_k / scale)
        self.trans_dim = trans_dim
        self.depth = depth
        self.num_group = num_groups
        self.group_size = group_size
        self.grid_size = grid_size  # Hilbert 网格大小

        # 1. 分组模块
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # 2. 局部特征编码
        self.encoder = Encoder(encoder_channel=self.trans_dim)

        # 3. 位置编码
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim)
        )

        # 4. Mamba 块
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]
        self.blocks = MixerModel(
            d_model=self.trans_dim,
            n_layer=self.depth,
            rms_norm=rms_norm,
            drop_out=drop_out,
            drop_path=dpr,
        )

        # 5. OrderScale 参数（双向扫描）
        self.OrderScale_gamma_1, self.OrderScale_beta_1 = init_OrderScale(
            self.trans_dim
        )
        self.OrderScale_gamma_2, self.OrderScale_beta_2 = init_OrderScale(
            self.trans_dim
        )

        # 6. 特征投影
        self.feature_proj = nn.Linear(self.trans_dim, self.dim_k)

        # 7. 聚合函数
        if sym_fn is None:
            sym_fn = _default_sym_fn
        self.sy = sym_fn

        # 兼容性属性
        self.t_out_h1 = None
        self.t_out_t2 = None

    def forward(self, points):
        """
        前向传播（带 Hilbert 序列化）

        Args:
            points: [B, N, 3] 输入点云

        Returns:
            features: [B, K] 全局特征向量
        """
        # 1. 分组 [B, N, 3] -> [B, G, S, 3], [B, G, 3]
        neighborhood, center = self.group_divider(points)

        # 2. 局部特征编码 [B, G, S, 3] -> [B, G, D]
        group_input_tokens = self.encoder(neighborhood)

        # 3. 位置编码 [B, G, 3] -> [B, G, D]
        pos = self.pos_embed(center)

        # 4. Hilbert 序列化 + 双向扫描
        from .serialization import serialization_func

        # 前向扫描（Hilbert）
        _, _, _, tokens_fwd, pos_fwd = serialization_func(
            center, group_input_tokens, pos, "hilbert", grid_size=self.grid_size
        )
        tokens_fwd = apply_OrderScale(
            tokens_fwd, self.OrderScale_gamma_1, self.OrderScale_beta_1
        )

        # 后向扫描（Hilbert 转置）
        _, _, _, tokens_bwd, pos_bwd = serialization_func(
            center, group_input_tokens, pos, "hilbert-trans", grid_size=self.grid_size
        )
        tokens_bwd = apply_OrderScale(
            tokens_bwd, self.OrderScale_gamma_2, self.OrderScale_beta_2
        )

        # 5. 拼接双向序列 [B, 2*G, D]
        x = torch.cat([tokens_fwd, tokens_bwd], dim=1)
        pos_cat = torch.cat([pos_fwd, pos_bwd], dim=1)

        # 6. Mamba 序列建模 [B, 2*G, D] -> [B, 2*G, D]
        x = self.blocks(x, pos_cat)

        # 保存中间特征（兼容性）
        self.t_out_h1 = x.transpose(1, 2)  # [B, D, 2*G]

        # 7. 全局聚合 [B, 2*G, D] -> [B, D]
        if hasattr(self.sy, "__name__") and "max" in self.sy.__name__:
            x = torch.max(x, dim=1)[0]
        else:
            x = self.sy(x)

        # 8. 投影到 dim_k [B, D] -> [B, K]
        x = self.feature_proj(x)

        return x

    def load_pretrained_weights(self, ckpt_path, strict=False, verbose=True):
        """
        加载 PointMamba 预训练权重

        Args:
            ckpt_path: 预训练权重文件路径
            strict: 是否严格匹配所有键
            verbose: 是否打印详细信息

        Returns:
            incompatible: 不兼容的键
            utilization_rate: 权重利用率（%）
        """
        if verbose:
            print(
                "[PointMamba Adapter Hilbert] Loading pretrained weights from "
                f"{ckpt_path}"
            )

        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "base_model" in ckpt:
            state_dict = ckpt["base_model"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # 清理键名
        cleaned_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            k = k.replace("MAE_encoder.", "")

            if "cls_head_finetune" in k:
                continue

            cleaned_dict[k] = v

        # 智能匹配
        adapter_dict = self.state_dict()
        matched_dict = {}
        unmatched_adapter = []

        for k, v in cleaned_dict.items():
            if k in adapter_dict:
                if v.shape == adapter_dict[k].shape:
                    matched_dict[k] = v
                else:
                    if verbose:
                        print(f"  ⚠️ Shape mismatch: {k}")
                        print(
                            "     Pretrained: "
                            f"{v.shape}, Adapter: {adapter_dict[k].shape}"
                        )

        for k in adapter_dict.keys():
            if k not in matched_dict:
                unmatched_adapter.append(k)

        # 加载权重
        incompatible = self.load_state_dict(matched_dict, strict=strict)

        # 计算利用率
        matched_params = sum(
            p.numel() for k, p in adapter_dict.items() if k in matched_dict
        )
        total_params = sum(p.numel() for p in adapter_dict.values())
        utilization_rate = matched_params / total_params * 100

        if verbose:
            print(
                "  ✅ Successfully loaded "
                f"{len(matched_dict)}/{len(adapter_dict)} parameters"
            )
            print(f"  📊 Weight utilization: {utilization_rate:.1f}%")
            print(f"     Matched params: {matched_params:,}/{total_params:,}")

            if unmatched_adapter:
                print(
                    "  ⚠️ "
                    f"{len(unmatched_adapter)} parameters need random initialization:"
                )
                for k in unmatched_adapter[:3]:
                    print(f"     - {k}")
                if len(unmatched_adapter) > 3:
                    print(f"     ... and {len(unmatched_adapter) - 3} more")

        return incompatible, utilization_rate


# ========== 分类器 ==========


class PointMamba_classifier(nn.Module):
    """PointMamba 分类器"""

    def __init__(self, num_c, ptfeat, dim_k):
        super().__init__()
        self.features = ptfeat

        self.classifier = nn.Sequential(
            nn.Linear(dim_k, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_c),
        )

    def forward(self, points):
        feat = self.features(points)
        out = self.classifier(feat)
        return out

    def loss(self, out, target, w=0.001):
        loss_c = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(out, dim=1), target, reduction="mean"
        )
        return loss_c


# ========== 辅助函数 ==========


def symfn_max(x):
    """最大池化聚合"""
    return torch.max(x, dim=1)[0]


def symfn_avg(x):
    """平均池化聚合"""
    return torch.mean(x, dim=1)


# ========== 测试代码 ==========

if __name__ == "__main__":
    print("=" * 60)
    print("PointMamba Adapter with Hilbert Serialization - Test")
    print("=" * 60)

    # 创建模型
    model = PointMamba_features(
        dim_k=1024,
        sym_fn=symfn_max,
        num_groups=64,
        group_size=32,
        trans_dim=384,
        depth=6,
        grid_size=0.02,
    )

    print("\n模型参数量:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 测试前向传播
    print("\n测试前向传播:")
    batch_size = 2
    num_points = 1024
    dummy_input = torch.randn(batch_size, num_points, 3)

    print(f"  Input shape: {dummy_input.shape}")

    # Mock the serialization import
    import sys
    from unittest.mock import MagicMock

    # Create a mock module
    mock_serialization = MagicMock()

    # Define the serialization_func on the mock
    def mock_serialization_func(center, group_input_tokens, pos, order, grid_size):
        return center, None, None, group_input_tokens, pos

    mock_serialization.serialization_func = mock_serialization_func
    sys.modules["ptlk.serialization"] = mock_serialization
    sys.modules["serialization"] = mock_serialization

    output = model(dummy_input)
    print(f"  Output shape: {output.shape}")
    assert output.shape == (batch_size, 1024), "Output shape mismatch!"
    print("  ✅ Forward pass successful!")

    # Cleanup mock
    del sys.modules["ptlk.serialization"]
    del sys.modules["serialization"]

    print("\n" + "=" * 60)
