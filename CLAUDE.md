# C3VD-Raycasting-Registration-Benchmark

面向医疗 cross-modal 点云配准的 benchmark-first 项目。当前目标是把仓库整理成一个基于 C3VD-raycasting 的多模型训练、测试与统一评测平台。

## Environment
```bash
source /home/linzhe/anaconda3/etc/profile.d/conda.sh
conda activate PCLR_compare
```

- Active project env name: `PCLR_compare`
- Benchmark train/eval, DPG-HPT scripts, and GeoTransformer/RegTR vendor bridges should be run in `PCLR_compare`, not the base Anaconda Python.
- If shell activation is unreliable on this machine, prefer calling the env interpreter directly:
  ```bash
  /home/linzhe/anaconda3/envs/PCLR_compare/bin/python
  ```
- For GeoTransformer on this machine, the most robust launch form is:
  ```bash
  LD_LIBRARY_PATH=/home/linzhe/anaconda3/envs/PCLR_compare/lib:/home/linzhe/anaconda3/envs/PCLR_compare/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \
  /home/linzhe/anaconda3/envs/PCLR_compare/bin/python
  ```
- Python `3.10.18`
- PyTorch `2.8.0+cu128`
- Open3D `0.19.0`
- NumPy `2.2.6` / SciPy `1.15.3` / h5py `3.14.0` / matplotlib `3.10.6` / PyYAML `6.0.2`
- `plyfile` is importable in the current environment
- Optional model/runtime deps available in the current environment: `mamba_ssm`, `knn_cuda`, `MinkowskiEngine`
- Optional tracking dependency available in the current environment: `wandb`
- Recent benchmark validation: Python compile checks and `23` benchmark tests passed during the three-table efficiency integration on `2026-04-13`
- Recent smoke validation: DCP/GeoTransformer token-stress smoke and GeoTransformer `train_benchmark.py` smoke completed on `2026-04-13`
- Known lint caveat: `ruff` is not installed in the active `PCLR_compare` environment
- Known conda caveat on this machine: activation may print a `conda-libmamba-solver` / `libarchive.so.20` warning, but the environment still activates successfully
- Practical runtime workaround for plotting/cache writes:
  ```bash
  MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp
  ```
- Tracking: wandb（可选，当前环境可初始化）

### Dataset Paths

- `C3VD`: `/mnt/f/Datasets/C3VD_sever_datasets`
- `C3VD_windows_source`: `F:\Datasets\C3VD_sever_datasets`
- `C3VD_Raycasting`: `/mnt/f/Datasets/C3VD_sever_datasets`
- `C3VDv2`: optional robustness/generalization extension root
- `Subset_Config`: `/home/linzhe/PCLR_compare/configs/subset_config.json`

## Tech Stack

- **Data generation:** depth reprojection + mesh raycasting + pose-conditioned pair generation
- **Benchmark runtime:** manifest-driven benchmark core + adapter/trainer bridges
- **Analysis runtime:** unified per-sample result schema + static report/export pipeline
- **Registered benchmark models:** ICP, DCP, PointNetLK, PointNetLK Revisited, Mamba3D, RegTR, GeoTransformer
- **Current focus:** 同一 C3VD 配置下完成 full train/eval 主榜，并补充 raw-point scalability 与 Transformer-like token stress
- **Env note:** PointNetLK_c3vd / Mamba 路线所需的 `mamba_ssm` 已可导入；RegTR/GeoTransformer 相关 sparse runtime 已完成本机接入

## Project Structure

```text
src/
  benchmarking/        stable benchmark core, including analysis/reporting subsystem
  common/              shared datasets / trainers / utils bridge layer
  unified_testing/     existing evaluation harness bridge layer
  PointNetLK_c3vd/     legacy research package
  PointMamba/          legacy external code
baselines/
  PointNetLK/
  PointNetLK_Revisited/
  dcp/
  RegTR/
  GeoTransformer/
configs/
  subset_config.json
  benchmark/           durable benchmark configs, including preprocess/analysis
scripts/
  benchmark/
  runners/             stable train/eval CLI entrypoints
docs/
  Project_Introduction.md
  Benchmark_Efficiency_Table_Design.md
tests/
  benchmark/           contract, analysis, and smoke tests
```

## Core Artifacts

- `PROJECT_STATE.json` — workflow state
- `iteration_log.json` — experiment history
- `project_map.json` — stable architecture map
- `.auto_iterate/` — controller-owned runtime state

## Entry Scripts

- Current stable seed script: `scripts/benchmark/build_c3vd_raycasting_manifest.py`
- Current legacy eval seed: `src/unified_testing/unified_test.py`
- Current stable benchmark CLI:
  - `scripts/runners/train_benchmark.py`
  - `scripts/runners/eval_benchmark.py`

## Stable Code Rule

- Read `project_map.json` before any non-trivial stable-code change.
- Update `project_map.json` when stable modules, interfaces, or benchmark contracts change.
- Treat `src/benchmarking/` as the future stable core; treat `src/common/` and `src/unified_testing/` as bridge layers.

## Workflow

WF1(survey) -> WF2(arch) -> WF3(check) -> WF4(data) -> WF5(baseline) -> WF6(plan) -> WF7(code) -> WF7.5(validate) -> WF8(iterate) -> WF9(final-exp) -> WF10(release)

Current stage: `WF7 (code_expert, completed)`

Project-direction gate: `CONDITIONAL GO`

Execution gate: `WF7 complete; post-WF7 benchmark hardening is active, but do not advance stage without explicit user confirmation`

Benchmark runtime policy:
- Stable benchmark train/eval runners use per-model runtime policy:
  - `ICP` is `benchmark-native` and must run with `runtime.device=cpu`
  - vendor-readonly bridged baselines require `runtime.device=cuda:*`
- In practice, vendor-readonly bridged baselines should be launched from the `PCLR_compare` Python 3.10 environment; base Python 3.12 on this machine is not a supported runtime for GeoTransformer and similar extensions.
- `baselines/*` are treated as readonly vendor subrepositories; tracked diffs are rejected for vendor-readonly models by benchmark runners

Current protocol decisions:
- Official comparisons must carry `preprocess_profile_id`
- `canonical_v1` is the default main-track preprocessing profile
- Sampling and perturbation are benchmark-owned and shared by train/eval
- Normalization is baseline-aware and model-private, but must be declared by registry/config
- Official result bundles must include multi-threshold success metrics, bucket reports, geometry diagnostics, efficiency/provenance fields, and failure analysis outputs
- `ResultRecord v2` is the analysis fact source for tables, curves, geometry summaries, and qualitative reports
- Static report bundles are a first-class v1 deliverable; heavy online dashboard infrastructure is not required for the first release
- Efficiency reporting is organized as three tables: main benchmark with training resources, raw-point memory scalability, and Transformer-like token stress

Important: do not auto-advance beyond the current workflow stage without explicit user confirmation.
Important: do not treat `C3VDv2` as a drop-in replacement for the `C3VD v1` core benchmark.
