# RegTR DPG-HPT Auditable Rationale Chain

- source: `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/proposal/agent_proposal.yaml`
- scope: extracted from explicit YAML fields only
- note: this is not a hidden chain-of-thought transcript; it is the retained rationale/evidence chain stored for reproducibility.

## 1. Firewall And Task Binding

- model: `regtr`
- target_dataset: `c3vd_raycasting_v1`
- used_official_test_feedback: `False`

## 2. Route Selection Rationale

### Selected Routes

#### `3DMatch`

RegTR 在本仓库暴露的唯一高置信 vendor route 是 3DMatch indoor metric scene route。它与 C3VD 都是 partial-to-partial scene/fragments 配准，并且默认保持 metric/raw-space 处理；相比 normalized-object 或 outdoor-lidar route，这条路线是 RegTR 迁移到 C3VD 的最可审计来源。

Evidence:
  - `baselines/RegTR/src/conf/3dmatch.yaml` :: `kpconv_options.first_subsampling_dl`
  - `baselines/RegTR/src/conf/3dmatch.yaml` :: `dataset.overlap_radius`
  - `baselines/RegTR/src/conf/3dmatch.yaml` :: `losses.r_p`
  - `baselines/RegTR/src/conf/3dmatch.yaml` :: `losses.r_n`
  - `baselines/RegTR/src/conf/3dmatch.yaml` :: `validation.reg_success_thresh_trans`

#### `c3vd_raycasting_v1`

目标画像显示 C3VD 是 medical endoscopic cross-modal partial-to-partial 数据集，raw coordinate 按 mm_like 处理，target bbox p50 约 122 raw units，source/target 最近邻间距 p50 约 0.85/0.87 raw units。因此迁移应保持 normalize_mode=none，并只缩放 RegTR 的尺度敏感半径/voxel 字段。

Evidence:
  - `src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json` :: `data`
  - `src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json` :: `geometry`

### Rejected Routes

#### `ModelNet40`

ModelNet40 是 normalized-object/full-to-full route，不符合 C3VD 的 medical endoscopic partial-to-partial scene route；RegTR 虽有 ModelNet config，但 route card 默认拒绝其作为 C3VD metric-scene 迁移来源。

Evidence:
  - `configs/benchmark/hparam_transfer/baseline_routes.yaml` :: `models.regtr.reject_for_c3vd_by_default`
  - `baselines/RegTR/src/conf/modelnet.yaml` :: `dataset.dataset`

#### `KITTI`

KITTI 是 outdoor lidar route，点密度、视野尺度和传感器模态均不同于 C3VD endoscopic local scene；RegTR route card 也没有暴露 KITTI vendor config route。

Evidence:
  - `configs/benchmark/hparam_transfer/baseline_routes.yaml` :: `models.regtr.reject_for_c3vd_by_default`

## 3. Target-Profile Interpretation

- dataset: `c3vd_raycasting_v1`
- domain: `medical_endoscopic_cross_modal`
- source_modality: `depth_reprojected_point_cloud`
- target_modality: `ct_mesh_raycasted_visible_point_cloud`
- pair_type: `partial_to_partial`
- coordinate_unit: `mm_like`
- inferred_unit: `mm_like`
- normalization_route: `none`
- point_count_profile: `{'source_p50': 32472.5, 'target_p50': 14572.5}`
- bbox_spacing_profile: `{'source_bbox_diag_p50': 107.09344100952148, 'target_bbox_diag_p50': 121.99967193603516, 'source_nn_spacing_p50': 0.8513259589672089, 'target_nn_spacing_p50': 0.869653195142746}`

## 4. Cross-Profile Route Comparison

### `3DMatch` / `indoor_metric_scene`

- status: `preferred`
- reason: 3DMatch 与 C3VD 都是 metric-space partial scene registration route， 且 RegTR 公开配置直接提供 KPConv voxel、overlap radius、feature positive/negative radii 和 validation translation threshold。差异在于 C3VD 的 raw coordinate 为 mm_like、局部视野更小且 cross-modal，因此只采用 route-scale 缩放，不改变 architecture、optimizer、loss family 或官方评测阈值。
- evidence:
  - `outputs/benchmark/hparam_transfer/regtr_measured_run/reference_profiles/regtr_reference_profiles.json` :: `routes[0].fields`
  - `src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json` :: `geometry`

## 5. Train-Hparam Decision Policy

- 使用 RegTR 实际暴露的 3DMatch metric-scene route 作为唯一迁移来源。
- C3VD 主榜 raw point budget 固定为 8192，归属 benchmark_owned。
- 保持 preprocess/profile raw metric route 与 RegTR private normalization route 均为 none/bn3_metric_unit。
- 将 3DMatch 长度类参数按 C3VD mm_like local-fragment scale 缩放；default 使用 100x，与现有 GeoTransformer C3VD metric-scene 迁移比例一致。
- 不改变 architecture、optimizer family、loss weights、训练 epoch/step 上限或官方 benchmark metric thresholds。

## 6. Default Candidate Field Decisions

| Field | Value | Owner/status | Transfer basis | Evidence |
| --- | --- | --- | --- | --- |
| `data.num_points` | `8192` | `benchmark_owned` | C3VD 主榜 raw point budget 是 benchmark-owned policy，不从 RegTR vendor config 迁移。 | `src/benchmarking/hparam_transfer/SKILL.md` :: `Main-Table Point Budget`<br>`docs/Benchmark_Efficiency_Table_Design.md` :: `primary main table raw point budget` |
| `preprocess.normalize_mode` | `none` | `model_private` | RegTR 的 3DMatch route 是 raw metric route；C3VD train/eval 均保持 none。 | `src/benchmarking/registry/model_registry.py` :: `default_eval_normalize_mode.regtr`<br>`src/benchmarking/bridges/configs/c3vd_regtr.yaml` :: `dataset.normalize_mode` |
| `model.private_normalization_route` | `bn3_metric_unit` | `model_private` | 保持 registry 声明的 RegTR metric-unit private route，不引入 unit_cube/object normalization。 | `src/benchmarking/registry/model_registry.py` :: `private_input_transform_id.regtr` |
| `data.voxel_size` | `2.5` | `transferred` | 3DMatch base voxel 0.025 乘 100x，得到 2.5 mm_like；沿用现有 metric-scene C3VD 迁移比例。 | `baselines/RegTR/src/conf/3dmatch.yaml` :: `kpconv_options.first_subsampling_dl`<br>`src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json` :: `geometry.nearest_neighbor_spacing`<br>`docs/Benchmark_Protocol_And_Hparam_Transfer_Summary.md` :: `GeoTransformer measured run transfer fields`<br>`src/benchmarking/runners/train_runner.py` :: `_apply_regtr_hparam_transfer_overrides` |
| `model.matching_radius` | `3.75` | `transferred` | 3DMatch overlap radius 0.0375 乘 100x，得到 3.75 mm_like，约为 C3VD NN p50 的 4.3x。 | `baselines/RegTR/src/conf/3dmatch.yaml` :: `dataset.overlap_radius`<br>`src/common/datasets/c3vd_for_regtr.py` :: `overlap_radius` |
| `losses.r_p` | `20.0` | `transferred` | 保持 RegTR 3DMatch 关系 r_p = 8 * first_subsampling_dl，随 default voxel 缩放到 20.0。 | `baselines/RegTR/src/conf/3dmatch.yaml` :: `losses.r_p`<br>`baselines/RegTR/src/conf/3dmatch.yaml` :: `kpconv_options.first_subsampling_dl` |
| `losses.r_n` | `40.0` | `transferred` | 保持 RegTR 3DMatch 关系 r_n = 16 * first_subsampling_dl，随 default voxel 缩放到 40.0。 | `baselines/RegTR/src/conf/3dmatch.yaml` :: `losses.r_n`<br>`baselines/RegTR/src/conf/3dmatch.yaml` :: `kpconv_options.first_subsampling_dl` |
| `eval.acceptance_radius` | `10.0` | `transferred` | 仅用于 RegTR checkpoint selection；0.1 乘 100x 得到 10mm-like，不改变 official rte threshold。 | `baselines/RegTR/src/conf/3dmatch.yaml` :: `validation.reg_success_thresh_trans` |

## 7. Candidate Scale Variants

- `conservative`: voxel=`2.0`, matching_radius=`3.0`, r_p/r_n=`16.0/32.0`, status=`transferred`
- `default`: voxel=`2.5`, matching_radius=`3.75`, r_p/r_n=`20.0/40.0`, status=`transferred`
- `aggressive`: voxel=`3.0`, matching_radius=`4.5`, r_p/r_n=`24.0/48.0`, status=`requires_user_approval`

## 8. Risks And Notes

### `regtr_c3vd_scale_factor_requires_validation`

- severity: `medium`
- note: 100x route-scale 来自 3DMatch metric-scene 到 C3VD mm_like local-fragment 的证据迁移，并沿用现有 GeoTransformer C3VD 迁移比例；它不是 test-set tuning，仍需要 train/val smoke 和正式 validation 证明。

### `regtr_no_native_3dlomatch_config`

- severity: `low`
- note: 3DLoMatch 可作为 low-overlap 风险参照，但 RegTR route card 只暴露 3DMatch config，因此候选不能由 3DLoMatch 单独驱动。

### `profile_source_manifest_path`

- severity: `low`
- note: durable dataset config 中的 /mnt/f manifest 当前不存在；本次上下文使用 outputs/benchmark/flow_check_eval/c3vd_raycasting_manifest.jsonl 重新生成 measured profile，并保留 profile-source config 以便审计。

Notes:

- 候选没有使用 official test feedback。
- aggressive candidate 中的 scale/radius 字段仅作为需要批准的探索项。
- promotion 只能在用户明确确认后执行。

## 9. Appendix Figure Use

For the agent-proposal panel, use Sections 2, 4, and 6 as the visible content: route decision, cross-profile comparison, and default-candidate field decisions. The full YAML can be cited as the retained model output.
