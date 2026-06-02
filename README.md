# C3VDReg

C3VDReg 是一个面向局部结肠镜点云配准的 benchmark-first 框架。它基于 C3VD 的 depth maps、camera poses 和 CT meshes 构造受控的 local-to-local registration 任务：source point cloud 来自结肠镜深度帧反投影，target point cloud 来自同一相机位姿下对 CT mesh 的 raycasting。

本仓库刻意只保留可复现框架。Git 中提交的是 benchmark code、configs、manifest snapshot、必要的 YAML run configs 和 checkpoint manifest；不提交 raw C3VD data、外部 baseline 源码、模型权重、论文生成物或运行输出。

## Benchmark 范围

本 README 按 BMVC draft 论文协议重写：`C3VDReg: Benchmarking Local Colonoscopic Registration for Anatomy-Aware Localization`。

C3VDReg 隔离的是 full local-to-global localization 之前的刚性 partial-to-partial alignment。raycast target 提供同一 C3VD frame 的 oracle visible CT surface，因此 benchmark 衡量的是方法能否把 video-depth source cloud 配准到对应的 visible CT target cloud。它不解决 full CT surface retrieval、nonrigid deformation、viewpoint mismatch、tissue motion、sequence tracking 或 clinical deployment。

主协议：

| 项目 | 设置 |
| --- | --- |
| Dataset | C3VD-derived raycasting pairs |
| Manifest | `manifests/c3vd_raycasting_v1.jsonl` |
| Pairs | 10,015 total: 5,952 train, 1,975 val, 2,088 test |
| Test scenes | `cecum_t1_a`, `cecum_t3_a`, `sigmoid_t3_b`, `trans_t1_a`, `trans_t2_b`, `trans_t4_a` |
| Input budget | 8,192 source points and 8,192 target points |
| Pose convention | source-to-target transform |
| Perturbation | R25-90deg/T100-500mm，只加在 source 上，无额外噪声 |
| Primary metrics | RR@5 at 5deg/5mm and RR@10 at 10deg/10mm |
| Error metrics | RRE in degrees, RTE in millimetres |
| Diagnostics | visible nearest-neighbor distance, trimmed Chamfer distance, latency |

论文结果说明这个受控任务仍未被当前公开方法解决。GeoTransformer 是最强 baseline，但只达到 17.43% RR@5 和 35.63% RR@10。测试集 GT overlap 很高，但 recall 随 overlap 仍低且非单调，说明主要瓶颈不是 shared surface 不足，而是重复管状解剖结构造成的 translation ambiguity。

## 仓库内容

当前提交内容：

- `src/`: benchmark runners、config schema、model registry、adapters、preprocessing、metrics、reporting 和 shared utilities。
- `scripts/`: train/eval entry points 和 benchmark analysis utilities。
- `configs/`: dataset、preprocess、model、runtime、hparam-transfer 和 paper-protocol YAML configs。
- `manifests/c3vd_raycasting_v1.jsonl`: portable manifest，内部路径相对 `data.dataset_root`。
- `outputs/**/*.yaml`: 只保留选中的 config snapshots。
- `SELECTED_CHECKPOINTS.csv`: checkpoint 目标路径、文件大小、SHA256 和下载链接占位。
- `requirements.txt`: Python dependency list。
- `README.md`: 唯一被 Git 跟踪的 Markdown 文档。

不提交内容：

- `baselines/*`
- `forks/PointNetLK_c3vd/`
- `checkpoints/*`
- `outputs/` 里的非 YAML 运行产物
- `BMVC2026_Linzhe/`
- raw C3VD data 和生成的 point clouds
- `*.ckpt`, `*.pt`, `*.pth`, `*.pth.tar`
- 除根目录 `README.md` 以外的所有 Markdown 文件

## 环境安装

从仓库根目录安装依赖：

```bash
git clone https://github.com/linzhe001/C3VDReg.git
cd C3VDReg
python -m pip install -r requirements.txt
```

如果使用 CUDA，请先安装与本机 CUDA driver 匹配的 PyTorch wheel。部分 vendor baselines，尤其是 RegTR 和 GeoTransformer，在 clone 源码后还可能需要各自的 compiled extensions 或额外环境步骤。

## 数据路径

提交的 manifest 使用相对路径，例如：

```text
C3VD_ply_source/<scene>/<frame>_depth_pcd.ply
visible_point_cloud_ply_depth/<scene>/frame_<frame>_visible.ply
C3VD_ref/<scene>/coverage_mesh.ply
```

当前 configs 默认数据根目录是：

```text
/mnt/f/Datasets/C3VD_sever_datasets
```

在其他机器上运行时，修改 YAML 里的 `data.dataset_root`，或复制一份 config 并写入本机 C3VD raycasting 数据根目录。不要把 raw data 提交进仓库。

## Baselines

外部 baseline 源码不进入本仓库。需要运行某个模型时，把对应 baseline clone 到 registry 或 adapter 期望的路径：

| Model id | Expected source path | Checkpoint |
| --- | --- | --- |
| `icp` | none | none |
| `dcp` | `baselines/dcp` | `checkpoints/dcp/model_best.pth` |
| `pointnetlk` | `baselines/PointNetLK` | `checkpoints/pointnetlk/c3vd_pointnetlk_model_model_best.pth` |
| `pointnetlk_revisited` | `baselines/PointNetLK_Revisited` | `checkpoints/pointnetlk_revisited/pointnetlk_c3vd_model_best.pth` |
| `mamba3d_mamba2_direct` | `forks/PointNetLK_c3vd` | `checkpoints/mamba3d_mamba2_direct/mamba3d_pointlk_model_best.pth` |
| `regtr` | `baselines/RegTR` | `checkpoints/regtr/model-142848.pth` |
| `geotransformer` | `baselines/GeoTransformer` | `checkpoints/geotransformer/geotransformer_c3vd_model_best.pth` |
| `bufferx` | `baselines/BUFFER-X` | configure separately |

这些 baseline 目录被 Git 忽略。clone 后把它们作为独立仓库维护，不要 vendor 到 C3VDReg 的 git history。

## Checkpoints

权重通过外部 artifact 分发。`SELECTED_CHECKPOINTS.csv` 记录了期望文件、大小和 SHA256。正式发布前，需要把里面的 `TBD` 替换成稳定下载链接。

推荐恢复流程：

```bash
mkdir -p checkpoints
rsync -av /path/to/C3VDReg_checkpoint/ checkpoints/
(cd checkpoints && sha256sum -c SHA256SUMS.txt)
```

公开发布时优先使用 Zenodo 或 Hugging Face Hub 这类带版本和校验能力的 artifact host。Google Drive 可以作为临时镜像，但不适合作为长期唯一来源。

## Evaluation

从仓库根目录运行 paper-protocol eval config：

```bash
python scripts/runners/eval_benchmark.py \
  --config configs/benchmark/paper_r25_90_t100_500mm/eval_geotransformer.yaml
```

运行全部 paper-protocol eval configs：

```bash
for cfg in configs/benchmark/paper_r25_90_t100_500mm/eval_*.yaml; do
  python scripts/runners/eval_benchmark.py --config "$cfg"
done
```

ICP 不需要 checkpoint。学习式 baseline 需要先准备对应的外部源码目录和 checkpoint path。

## Training

训练入口：

```bash
python scripts/runners/train_benchmark.py \
  --config configs/benchmark/runtime/smoke_train.yaml
```

正式训练可以从 `configs/benchmark/runtime/full_train.yaml` 开始，或复制 `outputs/**/config*.yaml` 里的现有 YAML snapshot，然后设置：

- `data.dataset_root`
- `model.id`
- `runtime.output_dir`
- model-specific overrides
- selected baseline 需要的 training overrides

训练日志、报告、表格、图片、checkpoints 和 per-sample predictions 应写入 `outputs/` 或 `checkpoints/`，并保持 untracked。

## 复现约束

主榜比较应保持同一个 source-to-target convention 和同一个 evaluation contract。核心文件是：

- `configs/benchmark/dataset/c3vd_raycasting_v1.yaml`
- `configs/benchmark/preprocess/canonical_v1.yaml`
- `configs/benchmark/paper_r25_90_t100_500mm/eval_*.yaml`
- `manifests/c3vd_raycasting_v1.jsonl`
- 与 `SELECTED_CHECKPOINTS.csv` 匹配的 checkpoint files

不要在不同 point budget、perturbation range、pose convention 或 private preprocessing 下直接比较方法；如果这样做，应明确标注为单独 diagnostic，而不是主协议结果。
