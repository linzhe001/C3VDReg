# Transfer Rules

## Sources
- `/home/linzhe/PCLR_compare/configs/benchmark/hparam_transfer/transfer_rules.yaml`
- `/home/linzhe/PCLR_compare/src/benchmarking/hparam_transfer/SKILL.md`

## transfer rules

```text
candidate_limit: 3
allowed_groups: length_like, density_like, normalization_route,
  runtime_sanity, heuristic_control
locked_groups: architecture, optimization, benchmark_policy
firewall: use_official_test_feedback=false
proposal_policy: require evidence, route rationale,
  cross-profile compatibility, reject locked/unknown fields
```

## rule file excerpt

```text
0001: schema_version: 1
0002: protocol_id: dataset_profile_guided_v1
0003: candidate_limit: 3
0004: 
0005: reference_selection:
0006:   score_terms:
0007:     - domain_match
0008:     - pair_type_match
0009:     - coordinate_route_match
0010:     - normalization_route_match
0011:     - baseline_native_support
0012:     - scale_profile_compatibility
0013:     - object_vs_scene_penalty
0014:     - outdoor_vs_endoscopic_penalty
0015:     - missing_config_source_penalty
0016:   preferred_reference_defaults:
0017:     dcp:
0018:       - ModelNet40
0019:     pointnetlk:
0020:       - ModelNet40
0021:     pointnetlk_revisited:
0022:       - 3DMatch
0023:     regtr:
0024:       - 3DMatch
0025:     geotransformer:
0026:       - 3DMatch
0027:     bufferx:
0028:       - 3DMatch
0029:       - Scannetpp_iphone
0030:     mamba3d:
0031:       - C3VD_PointNetLK_style
0032:       - ModelNet40
0033: 
0034: allowed_groups:
0035:   length_like:
0036:     - data.voxel_size
0037:     - data.voxel_size_0
0038:     - data.voxel_size_1
0039:     - data.downsample
0040:     - patch.des_r
0041:     - patch.search_radius_thresholds
0042:     - model.matching_radius
0043:     - model.patch_radius
0044:     - match.dist_th
0045:     - match.inlier_th
0046:     - match.kiss_resolution
0047:     - eval.acceptance_radius
0048:     - model.spatial_scale
0049:     - losses.r_p
0050:     - losses.r_n
0051:   density_like:
0052:     - data.max_numPts
0053:     - data.num_points
0054:     - patch.num_points_per_patch
0055:     - patch.num_fps
0056:     - patch.num_points_radius_estimate
0057:     - model.num_points_in_patch
0058:     - model.neighbor_limits
0059:   normalization_route:
0060:     - preprocess.normalize_mode
0061:     - model.private_normalization_route
0062:   runtime_sanity:
0063:     - runtime.batch_size
0064:     - runtime.num_workers
0065:     - runtime.memory_safe_point_budget
0066:   heuristic_control:
0067:     - heuristic_mode
0068:     - patch.fixed_des_radii
0069: 
0070: locked_groups:
0071:   architecture:
0072:     - model.backbone_depth
0073:     - model.hidden_dim
0074:     - model.embedding_dim
0075:     - model.attention_heads
0076:     - model.num_layers
0077:   optimization:
0078:     - train.optimizer
0079:     - train.loss_weights
0080:     - train.max_steps
0081:     - train.max_epochs
0082:   benchmark_policy:
0083:     - eval.rre_thresh
0084:     - eval.rte_thresh
0085:     - benchmark.primary_metric
```

## skill workflow excerpt

```text
0001: # Hparam Transfer
0002: 
0003: ## Purpose
0004: 
0005: This file is the stable repository-local guidance for `DPG-HPT` execution.
0006: Use it when converting a baseline into the benchmark-facing hyperparameter-transfer flow, or when producing:
0007: 
0008: - `reference_profiles`
0009: - `context_pack`
0010: - `agent_proposal`
0011: - `proposal_validation`
0012: - `candidate_configs`
0013: - `transfer_report`
0014: - `candidate_validation`
0015: - `promotion`
0016: 
0017: ## Required References
0018: 
0019: Read these before changing the stable protocol:
0020: 
0021: - `PROJECT_STATE.json`
0022: - `project_map.json`
0023: - `docs/DPG_HPT_Methodology.md`
0024: - `docs/Dataset_Profile_Guided_Hparam_Transfer_Plan.md`
0025: - `docs/Benchmark_Protocol_And_Hparam_Transfer_Summary.md`
0026: - `docs/Benchmark_Efficiency_Table_Design.md`
0027: 
0028: ## Core Rules
0029: 
0030: 1. Treat the protocol as `evidence-constrained configuration transfer`, not free-form tuning.
0031: 2. Keep architecture, optimizer family, benchmark metrics, and locked groups fixed unless the user explicitly changes the protocol.
0032: 3. Tie transferable parameters to concrete vendor config/code evidence.
0033: 4. Use measured target profiles when available; only fall back to durable profile stubs when the run is explicitly marked as fallback.
0034: 5. Treat unit semantics as a hard gate. Do not accept `declared_point_unit` blindly when raw geometry scale contradicts it.
0035: 6. Treat pose transform format, shape, direction, and translation unit as a hard gate. A profile is incomplete if it records geometry/unit statistics but omits the source pose field and the source-to-target vs target-to-source convention.
0036: 7. Build or refresh the common registration dataset census before model-specific transfer. The census must include C3VD raycasting and the common point-cloud registration datasets relevant to the baseline family.
0037: 8. Detect how the target baseline actually uses those common datasets before proposing transfers. The model route includes dataset family, domain, pair type, coordinate unit, normalization mode, private model normalization, point budget, voxel/radius scales, pose shape/direction, and evaluation radius.
0038: 9. Transfer candidates must be justified by both dataset-profile compatibility and model-specific route evidence. Do not hard-code a single source dataset as the transfer route.
0039: 10. Give 3DMatch/3DLoMatch special attention because many registration baselines expose metric-scene configs there, but use them as high-priority evidence within the dataset census, not as an unconditional default.
0040: 11. A C3VD train candidate is valid only after dataset-profile comparison, model-route profile generation, cross-profile compatibility analysis, and field-level train-hparam decisions are all recorded.
0041: 
0042: ## Main-Table Point Budget
0043: 
0044: The primary main-table raw point budget is a benchmark-owned policy, not a free transfer knob.
0045: 
0046: - Default primary main-table budget is `8192` raw points.
0047: - `12288` is only an optional high-resolution table, not the default main-table setting.
0048: - Do not reinterpret vendor `point_limit`, `num_points`, or similar density/count-like settings as permission to change the benchmark-wide main-table budget.
0049: - For main-table proposals, record `data.num_points = 8192` as `benchmark_owned` unless the user explicitly freezes a different benchmark-wide budget.
0050: 
0051: ## Unit and Perturbation Gate
0052: 
0053: Before proposing, training, evaluating, or repairing a result bundle:
0054: 
0055: - Verify `benchmark.point_unit`, manifest/profile unit, raw PLY coordinate scale, and metric conversion agree.
0056: - Verify the target profile `pose` section before using a candidate: source storage field, matrix shape, compact vs homogeneous representation, pose direction, translation unit, and SE(3) validity must all be recorded.
0057: - For C3VD, manifest `gt_transform` is the canonical benchmark pose field: it is a 4x4 homogeneous transform in raw coordinate units and follows the benchmark source-to-target contract used by metrics.
0058: - Baseline adapters must explicitly convert from the benchmark contract when vendor code expects a different pose shape or direction. For example, RegTR training consumes compact 3x4 `pose` tensors and must receive source-to-target transforms after perturbation handling.
0059: - For C3VD, current raw coordinates are `mm_like`: raw distance `x` should report as `x mm`, not `x * 1000 mm`.
0060: - Check physical plausibility with raw PLY bbox extent and pose translation scale. Do not choose a unit because it improves the score.
0061: - Record the evidence if a post-hoc unit repair is needed: scaled fields, scale factor, recomputed fields, backup location, and artifacts that were not regenerated.
0062: - Treat `translation_m` with care. The current perturbation implementation applies it directly in raw coordinate units. On `mm_like` C3VD, `translation_m=0.5` means about `0.5mm`, not `0.5m`.
0063: - Train/eval configs must align on `point_unit`, `preprocess.num_points_override`, `normalize_mode`, and perturbation. Any mismatch must be reported as a distribution shift.
0064: 
0065: ## Dataset Census, Model Route, and Normalization Gate
0066: 
0067: Before writing a `context_pack`, `agent_proposal`, `candidate_configs`, or `transfer_report`:
0068: 
0069: 1. Build the dataset census.
0070: 
0071:    - Inspect `src/benchmarking/hparam_transfer/dataset_profiles/dataset_profiles.yaml` and materialized profiles under `src/benchmarking/hparam_transfer/dataset_profiles/`.
0072:    - Include C3VD raycasting and every common point-cloud registration dataset that is already represented in the repo or mentioned by the target baseline's vendor configs/code. At minimum, check the existing registry entries such as `3DMatch`, `3DLoMatch`, `ModelNet40`, `KITTI`, and `ScanNet++_iPhone`; add evidence-backed stubs for other common datasets when the model uses them.
0073:    - For each dataset, record domain, source/target modality, pair type, coordinate unit, inferred unit, route hint, normalization route, raw/filtered point count, bbox diagonal, nearest-neighbor spacing, density/overlap hints when available, split policy, pose storage field, pose transform shape, pose direction, pose translation unit, perturbation unit, evaluation radius, and evidence source.
0074:    - Profile generation must inspect actual pose records, not only manifest metadata. For manifest-backed profiles, summarize transform shape counts, explicit-vs-defaulted transform counts, translation norm percentiles, rotation determinant/orthonormality checks, valid SE(3) fraction, and the code evidence that establishes direction semantics.
0075:    - Do not fabricate missing statistics. Mark unavailable fields as missing and keep the evidence status visible.
0076: 
0077: 2. Audit the target model's dataset routes.
0078: 
0079:    - Inspect `configs/benchmark/hparam_transfer/baseline_routes.yaml` for the baseline's known routes and preferred/rejected C3VD references.
0080:    - Inspect every route source listed there: vendor config/code when available, or the structured reference stub under `configs/benchmark/hparam_transfer/reference_configs/`.
0081:    - Search the baseline for additional dataset configs if the route card is incomplete. Add the discovered route to the proposal/report evidence before using it.
0082:    - For each model route, record the source dataset, route type, normalization, point budget, pose shape/direction convention, voxel/radius scales, patch/keypoint limits, acceptance radius, data augmentation, and the exact config/code paths that define them.
0083:    - Separate vendor-visible preprocessing from adapter-private transforms and benchmark-owned preprocessing. For example, record `preprocess.normalize_mode` separately from `model.private_normalization_route`.
0084: 
0085: 3. Transfer to C3VD by combining both sources of evidence.
0086: 
0087:    - First compare C3VD raycasting against the dataset census to identify compatible reference families by domain, modality, pair type, unit/scale, density, normalization route, and evaluation radius.
0088:    - Then intersect those compatible datasets with the target model's actual routes. A route cannot drive transfer if the baseline never used or exposed it.
0089:    - Prefer model-supported metric/local scene routes when C3VD compatibility supports them. `3DMatch`/`3DLoMatch` often deserve priority here, but they still need explicit model-route evidence.
0090:    - Treat `ModelNet40` `unit_cube`/normalized-object routes and `KITTI`/outdoor-lidar routes as mismatched for C3VD metric/endoscopic scene transfer unless a specific parameter is only density/runtime related or the mismatch is explicitly reported.
0091:    - Transfer only fields whose semantics are supported by the route evidence and allowlists, such as voxel size, matching radius, patch radius, point limit, `spatial_scale`, and acceptance radius.
0092:    - Keep the benchmark-owned main-table raw point budget at `8192` unless the user explicitly changes the benchmark policy.
0093: 
0094: 4. Apply the normalization gate.
0095: 
0096:    - For 3DMatch-like metric-scene routes, preserve metric-space assumptions by default: `normalize_mode: none` unless the vendor code explicitly applies another normalization.
0097:    - For ModelNet-like routes, treat `unit_cube`/normalized-object assumptions as a different operating route, not as a silent default for C3VD.
0098:    - Never infer normalization from score alone. A candidate must cite dataset-census compatibility and model-route evidence before `normalize_mode`, point budget, or radius-like fields are accepted.
0099:    - Train and eval must use the same effective normalization route. If train uses a bridge-specific `unit_cube` path while eval uses raw metric space, report the mismatch as invalid unless explicitly approved for an ablation.
0100: 
0101: ## Cross-Profile Compatibility and C3VD Train Hparams
0102: 
0103: Before accepting a C3VD training candidate, the `context_pack`, `agent_proposal`, or `transfer_report` must contain a cross-profile comparison that makes the route decision auditable.
0104: 
0105: The comparison must include:
0106: 
0107: - C3VD target profile fields: domain, source/target modality, pair type, coordinate unit, inferred unit, normalization route, point-count profile, bbox/spacing profile, split policy, pose storage/shape/direction/translation unit, perturbation unit, and evaluation/reporting unit.
0108: - Target-model route profiles from `reference_profiles`: every dataset route the model actually exposes, including source dataset, route type, normalization, point budget, pose shape/direction convention, scale/radius fields, patch/keypoint limits, data augmentation, acceptance radius, config path, and confidence.
0109: - Compatibility status for each model-supported route against C3VD: `preferred`, `usable_with_risk`, `density_only`, `runtime_only`, or `rejected`.
0110: - Rejection reasons for every unsupported or mismatched route, especially object-vs-scene, metric-vs-normalized, indoor-vs-outdoor, overlap regime, unit scale, and missing vendor evidence.
0111: 
0112: C3VD train-hparam decisions must then be derived from that comparison:
0113: 
0114: - Start from model-supported routes that are profile-compatible with C3VD. A dataset may be similar to C3VD, but it cannot drive transfer if the target model never exposes that route.
0115: - For each changed field, record `parameter_path`, chosen value, owner (`benchmark_owned`, `transferred`, `adapter_private`, `runtime_only`, or `locked`), source route/profile, conversion rule, and evidence path.
0116: - For main-table C3VD training, keep raw point budget at `8192` and record it as benchmark-owned unless the user explicitly changes the benchmark-wide policy.
0117: - Preserve C3VD `point_unit=mm_like` and require train/eval agreement on `preprocess.num_points_override`, `normalize_mode`, private normalization route, perturbation, and reporting metric units.
0118: - Preserve C3VD pose semantics: benchmark `gt_transform` is 4x4 source-to-target, and any baseline-private compact pose tensor or inverse transform must be documented as an adapter conversion rather than a dataset-profile change.
0119: - Transfer length-like parameters only when unit and scale compatibility are explicit. If the source route is `unit_cube`/ModelNet-like, do not transfer metric voxel/radius/acceptance thresholds into raw C3VD space without an explicit conversion and risk note.
0120: - Treat runtime settings such as batch size, workers, memory-safe point budget, and checkpoint resume separately from scientific hyperparameters.
```
