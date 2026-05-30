# PTLK - Point Cloud Registration Library with Mamba

**Point-cloud registration using Lucas-Kanade algorithm with multiple backbone architectures including Mamba-based models.**

## Overview

This library (`ptlk`) provides a modular implementation of point cloud registration algorithms, with support for various feature extraction backbones:

- **PointNet** - Original CVPR 2019 backbone
- **Mamba3D (v1-v4)** - State Space Model based backbones
- **PointMamba** - Advanced Mamba with Hilbert space-filling curve serialization
- **Attention** - Transformer-based backbone
- **CFormer** - Proxy-point based efficient attention
- **Fast Point Attention** - Lightweight attention variant

---

## Directory Structure

```
ptlk/
├── README.md                 # This file
├── __init__.py               # Module exports
│
├── # === Core Registration ===
├── pointlk.py                # PointNetLK algorithm (Lucas-Kanade optimization)
├── se3.py                    # SE(3) Lie group operations
├── so3.py                    # SO(3) Lie group operations
├── sinc.py                   # Sinc function utilities
├── invmat.py                 # Matrix inversion utilities
│
├── # === Feature Extractors (Backbones) ===
├── pointnet.py               # PointNet feature extractor
├── mamba3d_v1.py             # Mamba3D v1 - Pure PyTorch implementation
├── mamba3d_v2.py             # Mamba3D v2 - CUDA-accelerated (mamba-ssm)
├── mamba3d_v3.py             # Mamba3D v3 - With SE-Net attention
├── mamba3d_v4.py             # Mamba3D v4 - With CBAM attention
├── pointmamba_adapter.py     # PointMamba with Hilbert serialization
├── attention_v1.py           # Transformer-based backbone
├── fast_point_attention.py   # Lightweight attention
├── cformer.py                # CFormer (proxy-point attention)
│
├── # === PointMamba Dependencies ===
├── block_scan.py             # Mamba block with drop_path support
├── serialization.py          # Hilbert curve serialization
├── hilbert.py                # Hilbert curve utilities
│
├── # === Data Loading ===
├── data/
│   ├── __init__.py
│   ├── datasets.py           # Dataset classes (ModelNet, C3VD, etc.)
│   ├── transforms.py         # Point cloud transformations
│   ├── globset.py            # Glob-based dataset loading
│   └── mesh.py               # Mesh utilities
│
└── # === Other ===
    ├── dcp/                  # Deep Closest Point implementation
    └── adversarial.py        # Domain adversarial training (optional)
```

---

## Mamba3D Versions Comparison

| Version | Implementation | Features | Use Case |
|---------|---------------|----------|----------|
| **v1** | Pure PyTorch | Custom S6 layer, vectorized scan | CPU/Quick experiments |
| **v2** | mamba-ssm CUDA | High-performance SSM | Production/GPU training |
| **v3** | v2 + SE-Net | Channel attention (Squeeze-Excitation) | Better channel features |
| **v4** | v2 + CBAM | Channel + Spatial attention | Best feature representation |

### Mamba3D Architecture

```
Input: [B, N, 3] point cloud
    │
    ▼
┌─────────────────────────────┐
│  Input Projection (3 → D)   │  Linear layer
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  + Positional Encoding      │  Learnable [1, 2048, D]
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Mamba3DBlock × N           │
│  ┌───────────────────────┐  │
│  │ Mamba Layer           │  │  SSM with selective scan
│  │ (d_model, d_state,    │  │
│  │  d_conv=4, expand)    │  │
│  ├───────────────────────┤  │
│  │ LayerNorm             │  │
│  ├───────────────────────┤  │
│  │ FeedForward           │  │  Linear → GELU → Linear
│  │ (d_model → d_ff → D)  │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Feature Transform (MLP)    │  D → 256 → dim_k
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  Global Pooling             │  max / avg / selective
└─────────────────────────────┘
    │
    ▼
Output: [B, dim_k] global features
```

---

## Quick Start

### 1. Installation

```bash
# Core dependencies
pip install torch numpy scipy

# For Mamba3D v2/v3/v4 (CUDA required)
pip install mamba-ssm

# For PointMamba (optional)
pip install knn-cuda  # or use PyTorch fallback
```

### 2. Basic Usage

```python
import torch
from ptlk import mamba3d_v1, pointlk

# Create Mamba3D feature extractor
feature_extractor = mamba3d_v1.Mamba3D_features(
    dim_k=1024,           # Output feature dimension
    num_mamba_blocks=3,   # Number of Mamba blocks
    d_state=16,           # SSM state dimension
    expand=2              # Expansion factor
)

# Create PointNetLK model
model = pointlk.PointLK(
    ptnet=feature_extractor,
    delta=1.0e-2          # Finite difference step
)

# Forward pass
template = torch.randn(2, 1024, 3)  # [B, N, 3]
source = torch.randn(2, 1024, 3)

result = pointlk.PointLK.do_forward(
    model, template, source,
    maxiter=20,
    xtol=1e-7,
    p0_zero_mean=True,
    p1_zero_mean=True
)

# Get predicted transformation
g_pred = model.g  # [B, 4, 4] SE(3) matrix
```

### 3. Classification Task

```python
from ptlk import mamba3d_v2

# Create feature extractor
feat = mamba3d_v2.Mamba3D_features(
    dim_k=1024,
    num_mamba_blocks=3,
    d_state=16,
    expand=2
)

# Create classifier
classifier = mamba3d_v2.Mamba3D_classifier(
    num_c=40,        # Number of classes
    mambafeat=feat,
    dim_k=1024
)

# Forward pass
points = torch.randn(4, 1024, 3)  # [B, N, 3]
logits = classifier(points)       # [B, num_classes]

# Loss
target = torch.randint(0, 40, (4,))
loss = classifier.loss(logits, target)
```

### 4. Using PointMamba (Advanced)

```python
from ptlk import pointmamba_adapter

# Create PointMamba feature extractor
feat = pointmamba_adapter.PointMamba_features(
    dim_k=1024,
    num_groups=64,      # FPS groups
    group_size=32,      # Points per group
    trans_dim=384,      # Hidden dimension
    depth=6,            # Number of Mamba layers
    grid_size=0.02      # Hilbert grid size
)

# Load pretrained weights (optional)
feat.load_pretrained_weights('pretrain.pth', verbose=True)

# Forward pass
points = torch.randn(2, 1024, 3)
features = feat(points)  # [B, 1024]
```

---

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `dim_k` | Output feature dimension | 512, 1024 |
| `num_mamba_blocks` | Number of Mamba blocks | 1-4 |
| `d_state` | SSM state space dimension | 8, 16, 32 |
| `expand` | Internal expansion factor | 1.5, 2.0 |
| `d_model` | Hidden dimension | Auto: 128/scale |
| `sym_fn` | Aggregation function | max, avg, selective |

---

## Training Pipeline (Two-Stage)

### Stage 1: Classifier Pre-training

```bash
python train_classifier.py \
  --model-type mamba3d \
  --num-mamba-blocks 3 \
  --d-state 16 \
  --expand 2 \
  --dim-k 1024 \
  --epochs 100 \
  -o output/classifier \
  -i /path/to/dataset
```

### Stage 2: PointLK Fine-tuning

```bash
python train_pointlk.py \
  --model-type mamba3d \
  --transfer-from output/classifier_feat_best.pth \
  --pointnet tune \
  --epochs 200 \
  --mag 0.8 \
  --max-iter 20 \
  -o output/pointlk \
  -i /path/to/dataset
```

---

## API Reference

### `Mamba3D_features`

```python
class Mamba3D_features(nn.Module):
    def __init__(
        self,
        dim_k: int = 1024,           # Output feature dimension
        sym_fn: callable = symfn_max, # Aggregation function
        scale: int = 1,               # Scale factor (1=full, 2=half)
        num_mamba_blocks: int = 3,    # Number of Mamba blocks
        d_state: int = 16,            # SSM state dimension
        expand: float = 2             # Expansion factor
    ):
        ...

    def forward(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            points: [B, N, 3] input point cloud
        Returns:
            global_features: [B, dim_k]
            point_features: [B, N, dim_k]
        """
```

### `PointLK`

```python
class PointLK(nn.Module):
    def __init__(
        self,
        ptnet: nn.Module,    # Feature extractor
        delta: float = 1e-2  # Finite difference step
    ):
        ...

    @staticmethod
    def do_forward(
        model,
        p0: Tensor,          # Template [B, N, 3]
        p1: Tensor,          # Source [B, N, 3]
        maxiter: int = 10,   # Max LK iterations
        xtol: float = 1e-7,  # Convergence tolerance
        p0_zero_mean: bool = True,
        p1_zero_mean: bool = True
    ) -> Tensor:
        """Returns estimated transformation [B, 4, 4]"""
```

---

## Dependencies

### Required
- `torch >= 1.9.0`
- `numpy`
- `scipy`

### Optional (for advanced features)
- `mamba-ssm` - For Mamba3D v2/v3/v4 CUDA acceleration
- `knn-cuda` - For PointMamba KNN operations
- `open3d` - For point cloud I/O

---

## File Dependencies Graph

```
pointlk.py
├── se3.py
│   └── so3.py
│       └── sinc.py
├── invmat.py
└── [feature_extractor]
    ├── pointnet.py
    ├── mamba3d_v1.py (standalone)
    ├── mamba3d_v2.py ──► mamba-ssm
    ├── mamba3d_v3.py ──► mamba-ssm
    ├── mamba3d_v4.py ──► mamba-ssm
    └── pointmamba_adapter.py
        ├── block_scan.py
        ├── serialization.py
        └── hilbert.py
```

### Minimal Copy Set (Mamba3D v1 only)

If you only need Mamba3D v1 (pure PyTorch, no external dependencies):

```
ptlk/
├── __init__.py
├── pointlk.py
├── se3.py
├── so3.py
├── sinc.py
├── invmat.py
├── pointnet.py       # Optional fallback
└── mamba3d_v1.py     # Core Mamba implementation
```

### Full Copy Set (all Mamba variants)

```
ptlk/
├── __init__.py
├── pointlk.py
├── se3.py, so3.py, sinc.py, invmat.py
├── pointnet.py
├── mamba3d_v1.py, mamba3d_v2.py, mamba3d_v3.py, mamba3d_v4.py
├── pointmamba_adapter.py
├── block_scan.py, serialization.py, hilbert.py
└── data/  (if you need dataset classes)
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{aoki2019pointnetlk,
  title={PointNetLK: Robust & Efficient Point Cloud Registration Using PointNet},
  author={Aoki, Yasuhiro and Goforth, Hunter and Srivatsan, Rangaprasad Arun and Lucey, Simon},
  booktitle={CVPR},
  year={2019}
}

@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

---

## License

MIT License - See LICENSE file for details.
