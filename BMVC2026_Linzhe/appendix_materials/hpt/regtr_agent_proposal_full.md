# RegTR DPG-HPT Agent Proposal YAML

- source: `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/proposal/agent_proposal.yaml`
- model: `regtr`
- target_dataset: `c3vd_raycasting_v1`
- agent: `codex`
- run_id: `regtr_dpg_hpt_c3vd_20260430`
- used_official_test_feedback: `False`

```yaml
schema_version: 1
model: regtr
target_dataset: c3vd_raycasting_v1
agent:
  name: codex
  run_id: regtr_dpg_hpt_c3vd_20260430
test_set_firewall:
  used_official_test_feedback: false
reference_selection:
  selected:
  - dataset: 3DMatch
    reason: RegTR 在本仓库暴露的唯一高置信 vendor route 是 3DMatch indoor metric scene route。它与 C3VD 都是 partial-to-partial
      scene/fragments 配准，并且默认保持 metric/raw-space 处理；相比 normalized-object 或 outdoor-lidar route，这条路线是 RegTR
      迁移到 C3VD 的最可审计来源。
    evidence:
    - path: baselines/RegTR/src/conf/3dmatch.yaml
      field: kpconv_options.first_subsampling_dl
    - path: baselines/RegTR/src/conf/3dmatch.yaml
      field: dataset.overlap_radius
    - path: baselines/RegTR/src/conf/3dmatch.yaml
      field: losses.r_p
    - path: baselines/RegTR/src/conf/3dmatch.yaml
      field: losses.r_n
    - path: baselines/RegTR/src/conf/3dmatch.yaml
      field: validation.reg_success_thresh_trans
  - dataset: c3vd_raycasting_v1
    reason: 目标画像显示 C3VD 是 medical endoscopic cross-modal partial-to-partial 数据集，raw coordinate 按 mm_like
      处理，target bbox p50 约 122 raw units，source/target 最近邻间距 p50 约 0.85/0.87 raw units。因此迁移应保持 normalize_mode=none，并只缩放
      RegTR 的尺度敏感半径/voxel 字段。
    evidence:
    - path: src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json
      field: data
    - path: src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json
      field: geometry
  rejected:
  - dataset: ModelNet40
    reason: ModelNet40 是 normalized-object/full-to-full route，不符合 C3VD 的 medical endoscopic partial-to-partial
      scene route；RegTR 虽有 ModelNet config，但 route card 默认拒绝其作为 C3VD metric-scene 迁移来源。
    evidence:
    - path: configs/benchmark/hparam_transfer/baseline_routes.yaml
      field: models.regtr.reject_for_c3vd_by_default
    - path: baselines/RegTR/src/conf/modelnet.yaml
      field: dataset.dataset
  - dataset: KITTI
    reason: KITTI 是 outdoor lidar route，点密度、视野尺度和传感器模态均不同于 C3VD endoscopic local scene；RegTR route card
      也没有暴露 KITTI vendor config route。
    evidence:
    - path: configs/benchmark/hparam_transfer/baseline_routes.yaml
      field: models.regtr.reject_for_c3vd_by_default
cross_profile_compatibility:
  target_profile:
    dataset: c3vd_raycasting_v1
    digest: 05359f2115c9dde87e972b75d55f3e458eb522f5e994d3c2432182b3bd79d45a
    domain: medical_endoscopic_cross_modal
    source_modality: depth_reprojected_point_cloud
    target_modality: ct_mesh_raycasted_visible_point_cloud
    pair_type: partial_to_partial
    coordinate_unit: mm_like
    inferred_unit: mm_like
    normalization_route: none
    point_count_profile:
      source_p50: 32472.5
      target_p50: 14572.5
    bbox_spacing_profile:
      source_bbox_diag_p50: 107.09344100952148
      target_bbox_diag_p50: 121.99967193603516
      source_nn_spacing_p50: 0.8513259589672089
      target_nn_spacing_p50: 0.869653195142746
    split_policy: scene_safe
    perturbation_unit: raw_coordinate_units_mm_like
    reporting_unit: millimeter
  route_comparisons:
  - dataset: 3DMatch
    route: indoor_metric_scene
    status: preferred
    reason: 3DMatch 与 C3VD 都是 metric-space partial scene registration route， 且 RegTR 公开配置直接提供 KPConv voxel、overlap
      radius、feature positive/negative radii 和 validation translation threshold。差异在于 C3VD 的 raw coordinate
      为 mm_like、局部视野更小且 cross-modal，因此只采用 route-scale 缩放，不改变 architecture、optimizer、loss family 或官方评测阈值。
    evidence:
    - path: outputs/benchmark/hparam_transfer/regtr_measured_run/reference_profiles/regtr_reference_profiles.json
      field: routes[0].fields
    - path: src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json
      field: geometry
  train_hparam_decision_policy:
  - 使用 RegTR 实际暴露的 3DMatch metric-scene route 作为唯一迁移来源。
  - C3VD 主榜 raw point budget 固定为 8192，归属 benchmark_owned。
  - 保持 preprocess/profile raw metric route 与 RegTR private normalization route 均为 none/bn3_metric_unit。
  - 将 3DMatch 长度类参数按 C3VD mm_like local-fragment scale 缩放；default 使用 100x，与现有 GeoTransformer C3VD metric-scene
    迁移比例一致。
  - 不改变 architecture、optimizer family、loss weights、训练 epoch/step 上限或官方 benchmark metric thresholds。
review_queue:
- data.num_points
- preprocess.normalize_mode
- model.private_normalization_route
- data.voxel_size
- model.matching_radius
- losses.r_p
- losses.r_n
- eval.acceptance_radius
candidates:
  conservative:
    params:
      data.num_points:
        value: 8192
        status: benchmark_owned
        evidence:
        - path: src/benchmarking/hparam_transfer/SKILL.md
          field: Main-Table Point Budget
        - path: docs/Benchmark_Efficiency_Table_Design.md
          field: primary main table raw point budget
        transfer_basis: C3VD 主榜 raw point budget 是 benchmark-owned policy，不从 RegTR vendor config 迁移。
      preprocess.normalize_mode:
        value: none
        status: model_private
        evidence:
        - path: src/benchmarking/registry/model_registry.py
          field: default_eval_normalize_mode.regtr
        - path: src/benchmarking/bridges/configs/c3vd_regtr.yaml
          field: dataset.normalize_mode
        transfer_basis: RegTR 的 3DMatch route 是 raw metric route；C3VD train/eval 均保持 none。
      model.private_normalization_route:
        value: bn3_metric_unit
        status: model_private
        evidence:
        - path: src/benchmarking/registry/model_registry.py
          field: private_input_transform_id.regtr
        transfer_basis: 保持 registry 声明的 RegTR metric-unit private route，不引入 unit_cube/object normalization。
      data.voxel_size:
        value: 2.0
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: kpconv_options.first_subsampling_dl
        - path: src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json
          field: geometry.nearest_neighbor_spacing
        - path: src/benchmarking/runners/train_runner.py
          field: _apply_regtr_hparam_transfer_overrides
        transfer_basis: 3DMatch base voxel 0.025 乘 80x，得到 2.0 mm_like，约为 C3VD source/target NN p50 的 2.3x。
      model.matching_radius:
        value: 3.0
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: dataset.overlap_radius
        - path: src/common/datasets/c3vd_for_regtr.py
          field: overlap_radius
        transfer_basis: 3DMatch overlap radius 0.0375 乘 80x，作为较保守的 C3VD overlap/correspondence radius。
      losses.r_p:
        value: 16.0
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: losses.r_p
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: kpconv_options.first_subsampling_dl
        transfer_basis: 保持 RegTR 3DMatch 关系 r_p = 8 * first_subsampling_dl，随 conservative voxel 缩放到 16.0。
      losses.r_n:
        value: 32.0
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: losses.r_n
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: kpconv_options.first_subsampling_dl
        transfer_basis: 保持 RegTR 3DMatch 关系 r_n = 16 * first_subsampling_dl，随 conservative voxel 缩放到 32.0。
      eval.acceptance_radius:
        value: 8.0
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: validation.reg_success_thresh_trans
        transfer_basis: 仅用于 RegTR checkpoint selection 的 translation success threshold；0.1 乘 80x 得到 8mm-like，不改变官方
          benchmark metric thresholds。
  default:
    params:
      data.num_points:
        value: 8192
        status: benchmark_owned
        evidence:
        - path: src/benchmarking/hparam_transfer/SKILL.md
          field: Main-Table Point Budget
        - path: docs/Benchmark_Efficiency_Table_Design.md
          field: primary main table raw point budget
        transfer_basis: C3VD 主榜 raw point budget 是 benchmark-owned policy，不从 RegTR vendor config 迁移。
      preprocess.normalize_mode:
        value: none
        status: model_private
        evidence:
        - path: src/benchmarking/registry/model_registry.py
          field: default_eval_normalize_mode.regtr
        - path: src/benchmarking/bridges/configs/c3vd_regtr.yaml
          field: dataset.normalize_mode
        transfer_basis: RegTR 的 3DMatch route 是 raw metric route；C3VD train/eval 均保持 none。
      model.private_normalization_route:
        value: bn3_metric_unit
        status: model_private
        evidence:
        - path: src/benchmarking/registry/model_registry.py
          field: private_input_transform_id.regtr
        transfer_basis: 保持 registry 声明的 RegTR metric-unit private route，不引入 unit_cube/object normalization。
      data.voxel_size:
        value: 2.5
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: kpconv_options.first_subsampling_dl
        - path: src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json
          field: geometry.nearest_neighbor_spacing
        - path: docs/Benchmark_Protocol_And_Hparam_Transfer_Summary.md
          field: GeoTransformer measured run transfer fields
        - path: src/benchmarking/runners/train_runner.py
          field: _apply_regtr_hparam_transfer_overrides
        transfer_basis: 3DMatch base voxel 0.025 乘 100x，得到 2.5 mm_like；沿用现有 metric-scene C3VD 迁移比例。
      model.matching_radius:
        value: 3.75
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: dataset.overlap_radius
        - path: src/common/datasets/c3vd_for_regtr.py
          field: overlap_radius
        transfer_basis: 3DMatch overlap radius 0.0375 乘 100x，得到 3.75 mm_like，约为 C3VD NN p50 的 4.3x。
      losses.r_p:
        value: 20.0
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: losses.r_p
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: kpconv_options.first_subsampling_dl
        transfer_basis: 保持 RegTR 3DMatch 关系 r_p = 8 * first_subsampling_dl，随 default voxel 缩放到 20.0。
      losses.r_n:
        value: 40.0
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: losses.r_n
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: kpconv_options.first_subsampling_dl
        transfer_basis: 保持 RegTR 3DMatch 关系 r_n = 16 * first_subsampling_dl，随 default voxel 缩放到 40.0。
      eval.acceptance_radius:
        value: 10.0
        status: transferred
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: validation.reg_success_thresh_trans
        transfer_basis: 仅用于 RegTR checkpoint selection；0.1 乘 100x 得到 10mm-like，不改变 official rte threshold。
  aggressive:
    params:
      data.num_points:
        value: 8192
        status: benchmark_owned
        evidence:
        - path: src/benchmarking/hparam_transfer/SKILL.md
          field: Main-Table Point Budget
        - path: docs/Benchmark_Efficiency_Table_Design.md
          field: primary main table raw point budget
        transfer_basis: C3VD 主榜 raw point budget 是 benchmark-owned policy，不从 RegTR vendor config 迁移。
      preprocess.normalize_mode:
        value: none
        status: model_private
        evidence:
        - path: src/benchmarking/registry/model_registry.py
          field: default_eval_normalize_mode.regtr
        - path: src/benchmarking/bridges/configs/c3vd_regtr.yaml
          field: dataset.normalize_mode
        transfer_basis: RegTR 的 3DMatch route 是 raw metric route；C3VD train/eval 均保持 none。
      model.private_normalization_route:
        value: bn3_metric_unit
        status: model_private
        evidence:
        - path: src/benchmarking/registry/model_registry.py
          field: private_input_transform_id.regtr
        transfer_basis: 保持 registry 声明的 RegTR metric-unit private route，不引入 unit_cube/object normalization。
      data.voxel_size:
        value: 3.0
        status: requires_user_approval
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: kpconv_options.first_subsampling_dl
        - path: src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json
          field: geometry.nearest_neighbor_spacing
        - path: src/benchmarking/runners/train_runner.py
          field: _apply_regtr_hparam_transfer_overrides
        transfer_basis: 3DMatch base voxel 0.025 乘 120x，得到更粗的 3.0 mm_like hierarchy；需要用户批准后才适合作为主候选。
      model.matching_radius:
        value: 4.5
        status: requires_user_approval
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: dataset.overlap_radius
        - path: src/common/datasets/c3vd_for_regtr.py
          field: overlap_radius
        transfer_basis: 3DMatch overlap radius 0.0375 乘 120x；更宽 overlap gate 可能提高 correspondence recall，但会增加错误
          overlap 风险。
      losses.r_p:
        value: 24.0
        status: requires_user_approval
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: losses.r_p
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: kpconv_options.first_subsampling_dl
        transfer_basis: 保持 r_p = 8 * first_subsampling_dl，随 aggressive voxel 缩放到 24.0。
      losses.r_n:
        value: 48.0
        status: requires_user_approval
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: losses.r_n
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: kpconv_options.first_subsampling_dl
        transfer_basis: 保持 r_n = 16 * first_subsampling_dl，随 aggressive voxel 缩放到 48.0。
      eval.acceptance_radius:
        value: 12.0
        status: requires_user_approval
        evidence:
        - path: baselines/RegTR/src/conf/3dmatch.yaml
          field: validation.reg_success_thresh_trans
        transfer_basis: 仅用于 RegTR checkpoint selection；更宽 12mm-like 阈值可能偏宽，因此需要用户批准。
risks:
- id: regtr_c3vd_scale_factor_requires_validation
  severity: medium
  note: 100x route-scale 来自 3DMatch metric-scene 到 C3VD mm_like local-fragment 的证据迁移，并沿用现有 GeoTransformer
    C3VD 迁移比例；它不是 test-set tuning，仍需要 train/val smoke 和正式 validation 证明。
- id: regtr_no_native_3dlomatch_config
  severity: low
  note: 3DLoMatch 可作为 low-overlap 风险参照，但 RegTR route card 只暴露 3DMatch config，因此候选不能由 3DLoMatch 单独驱动。
- id: profile_source_manifest_path
  severity: low
  note: durable dataset config 中的 /mnt/f manifest 当前不存在；本次上下文使用 outputs/benchmark/flow_check_eval/c3vd_raycasting_manifest.jsonl
    重新生成 measured profile，并保留 profile-source config 以便审计。
notes_for_agent:
- 候选没有使用 official test feedback。
- aggressive candidate 中的 scale/radius 字段仅作为需要批准的探索项。
- promotion 只能在用户明确确认后执行。
digests:
  context_digest: e66947de47bdb84750047b0b1432e214701b50ce4e9927459fd2d86989899b09
  template_digest: null
```
