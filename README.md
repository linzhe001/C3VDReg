# C3VDReg

C3VDReg 是一个面向 C3VD raycasting local-to-local point cloud registration 的 benchmark-first 训练与评测框架。仓库只保留可复现框架、稳定配置、manifest 和必要说明；外部 baseline 源码、运行输出和模型权重不进入 git 历史。

## 提取依据

正文主协议为 R25-90deg/T100-500mm source-only perturbation：

- 数据集：C3VD raycasting local-to-local registration。
- 测试集：2088 个 held-out test pairs。
- 输入点数：source/target 均为 8192 points。
- 预处理：`canonical_v1`。
- 训练入口：`scripts/runners/train_benchmark.py`。
- 评测入口：`scripts/runners/eval_benchmark.py`。
- 主榜模型：`GeoTransformer`, `ICP`, `RegTR`, `PointNetLK-Mamba`, `PointNetLK Revisited`, `PointNetLK`, `DCP`。

## 目录内容

- `src/`: benchmark core、adapter、runner、reporting 和 shared dataset/utils 代码。
- `scripts/`: manifest 构建、训练/评测入口、paper figure/export 和诊断脚本。
- `configs/`: benchmark dataset/preprocess/model/runtime 配置，以及可直接运行的 paper eval configs。
- `manifests/`: 可提交的 C3VD raycasting manifest snapshot，路径相对 `data.dataset_root`。
- `baselines/`: 只提交 `README.md`；外部 baseline 源码由用户 clone 到固定路径。
- `forks/`: 只提交 `README.md`；`forks/PointNetLK_c3vd/` 由用户 clone。
- `checkpoints/`: 本地权重缓存目录，git 忽略；按 `SELECTED_CHECKPOINTS.csv` 校验。
- `outputs/`: 运行时输出目录，git 忽略；仅允许提交其中的 YAML config snapshot。
- `SELECTED_CHECKPOINTS.csv`: paper eval 使用的 checkpoint 清单、大小、SHA256 和下载链接占位。
- `CHECKPOINTS.md`: checkpoint 托管建议和恢复方式。

## Baseline 和 checkpoint 边界

本仓库不提交 baseline 源码。用户需要把外部 baseline clone 到 `baselines/README.md` 记录的固定路径；runner 会检查 vendor baseline 是独立 git repo 且没有 tracked diff。

权重也不建议提交到普通 git。推荐最终用 Zenodo 或 Hugging Face Hub 发布一个版本化 checkpoint bundle；Google Drive 可以作为临时共享镜像。每个 checkpoint 的目标位置以 `SELECTED_CHECKPOINTS.csv` 和 `configs/benchmark/paper_r25_90_t100_500mm/eval_*.yaml` 中的 `model.checkpoint_path` 为准。

ICP 是 classical CPU baseline，没有模型权重。

## 运行方式

从本目录运行 portable eval config：

```bash
cd /home/linzhe/PCLR_compare/C3VDReg
source /home/linzhe/anaconda3/etc/profile.d/conda.sh
conda activate PCLR_compare

python scripts/runners/eval_benchmark.py \
  --config configs/benchmark/paper_r25_90_t100_500mm/eval_geotransformer.yaml
```

训练入口同样从本目录运行，例如：

```bash
python scripts/runners/train_benchmark.py \
  --config configs/benchmark/runtime/smoke_train.yaml
```

`configs/benchmark/paper_r25_90_t100_500mm/` 中的 eval config 使用 repo-relative manifest 和 checkpoint path；运行后产生的报告、表格、图片和 per-sample result 会写入 `outputs/`，默认不提交。

## 数据边界

本目录没有复制 C3VD 原始 PLY/depth/mesh 数据。portable configs 默认仍指向本机数据根：

```text
/mnt/f/Datasets/C3VD_sever_datasets
```

在其他机器上运行时，只需要把 config 里的 `data.dataset_root` 改到可用的 C3VD raycasting 数据根。`manifests/c3vd_raycasting_v1.jsonl` 内部使用相对路径，不绑定本机 `/mnt/f`。

## 提交边界

- 提交：`src/`, `scripts/`, `configs/`, `manifests/`, `README.md`, `CHECKPOINTS.md`, `SELECTED_CHECKPOINTS.csv`, `baselines/README.md`, `forks/README.md`, `checkpoints/README.md`。
- 不提交：`baselines/*` 源码、`forks/PointNetLK_c3vd/`、`checkpoints/*` 权重、`outputs/` 非 YAML 运行产物、raw C3VD 数据。
- 如需保留历史结果，用外部 artifact bundle 或 release asset 保存，不放入主 git 历史。
