# C3VDReg benchmark extraction

本目录是从 `BMVC2026_Linzhe/bmvc_c3vd_benchmark_draft.tex` 对应实验中提取的 C3VDReg benchmark 代码、配置、论文材料、评测输出和正文实际使用的模型权重。

## 提取依据

正文主协议为 R25-90deg/T100-500mm source-only perturbation：

- 数据集：C3VD raycasting local-to-local registration。
- 测试集：2088 个 held-out test pairs。
- 输入点数：source/target 均为 8192 points。
- 预处理：`canonical_v1`。
- 评测入口：`scripts/runners/eval_benchmark.py`。
- 主榜模型：`GeoTransformer`, `ICP`, `RegTR`, `PointNetLK-Mamba`, `PointNetLK Revisited`, `PointNetLK`, `DCP`。
- 主榜结果来源：`outputs/benchmark/r25_90_t100_500mm_protocol/combined_leaderboard.md`。

## 目录内容

- `src/`: benchmark core、adapter、runner、reporting 和 shared dataset/utils 代码。
- `scripts/`: manifest 构建、训练/评测入口、paper figure/export 和诊断脚本。
- `configs/`: 原项目 benchmark 配置，以及可从本目录运行的 portable paper eval configs。
- `baselines/`: 正文模型需要的 vendor/fork 代码：`dcp`, `RegTR`, `GeoTransformer`, `PointNetLK`, `PointNetLK_Revisited`；vendor 自带示例权重已移除。
- `forks/PointNetLK_c3vd/`: `PointNetLK-Mamba` / `mamba3d_mamba2_direct` 所需代码；历史 ModelNet 权重已移除。
- `outputs/benchmark/r25_90_t100_500mm_protocol/`: 正文主协议的 eval configs、summary、leaderboard、per-sample results、error analysis 和 figures。
- `outputs/benchmark/flow_check_eval/`: 主协议 eval config 使用的 manifest 和 smoke config snapshot。
- `outputs/benchmark/**/train/`: 只保留正文 eval config 实际指向的 checkpoint 及最小训练元数据。
- `BMVC2026_Linzhe/`: 论文草稿、图片和 appendix/source material snapshot。
- `SELECTED_CHECKPOINTS.csv`: 正文实际使用的 checkpoint 清单、大小和 SHA256。

## 权重选择

权重选择以 `outputs/benchmark/r25_90_t100_500mm_protocol/configs/eval_*.yaml` 中的 `model.checkpoint_path` 为准。只复制这些 checkpoint：

- GeoTransformer: `outputs/benchmark/r90_t500mm_protocol/geotransformer/train/train_bridge/geotransformer_c3vd_model_best.pth`
- RegTR: `outputs/benchmark/r90_t500mm_from_scratch_fixed_regtr_dcp/regtr/train/train_bridge_logs/c3vd/260521_105941_benchmark_full_regtr/ckpt/model-142848.pth`
- PointNetLK-Mamba: `outputs/benchmark/mamba2_followup_point_order_pair_initializer/direct_sort_xyz_e5/train/train_bridge/mamba3d_pointlk_model_best.pth`
- PointNetLK Revisited: `outputs/benchmark/r90_t500mm_protocol/pointnetlk_revisited/train/train_bridge/pointnetlk_c3vd_model_best.pth`
- PointNetLK: `outputs/benchmark/r90_t500mm_protocol/pointnetlk/train/train_bridge/c3vd_pointnetlk_model_model_best.pth`
- DCP: `outputs/benchmark/r90_t500mm_from_scratch_fixed_regtr_dcp/dcp/train/train_bridge/models/model_best.pth`

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

`configs/benchmark/paper_r25_90_t100_500mm/` 中的配置已经把原始 `/home/linzhe/PCLR_compare/` 前缀改为 package-relative path。原始绝对路径配置仍保留在 `outputs/benchmark/r25_90_t100_500mm_protocol/configs/`。

## 数据边界

本目录没有复制 C3VD 原始 PLY/depth/mesh 数据。portable configs 默认仍指向本机数据根：

```text
/mnt/f/Datasets/C3VD_sever_datasets
```

在其他机器上运行时，需要把 config 里的 `data.dataset_root` 改到可用的 C3VD raycasting 数据根。

## 未复制内容

没有复制整个 `outputs/benchmark`，因为其中包含旧协议、smoke run、消融、中间 epoch 和未进入正文主表的权重。正文主表之外的 Buffer-X、method-level validation 和历史 publication package 权重/运行产物也未纳入本提取包。
