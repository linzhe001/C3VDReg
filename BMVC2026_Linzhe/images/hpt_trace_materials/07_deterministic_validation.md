# Deterministic Validation

## Sources
- `/home/linzhe/PCLR_compare/src/benchmarking/hparam_transfer/proposal_validation.py`
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/validated/proposal_validation.json`
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/validated/transfer_trace.json`

## deterministic validation pseudo-code

```text
proposal = load_agent_proposal(agent_proposal.yaml)
context = load_context_pack(context_pack.json)
check schema_version, model, target_dataset
require used_official_test_feedback == false
require candidate_count <= candidate_limit
validate selected/rejected route rationales
validate cross-profile route comparisons + evidence
for each candidate field:
  reject locked fields and fields outside allowlist
  require allowed owner/status label
  require evidence path and transfer_basis
if passed: normalize candidate configs and write trace digests
```

## RegTR validation output

```text
passed: True
errors: 0
warnings: 5
candidate_count: 3
validated_candidates: ['aggressive', 'conservative', 'default']
context_digest: e66947de47bdb847...
proposal_digest: d7967f079768a400...
candidate_bundle_digest: e8bbeee0bf3b116d...
```

## validation JSON excerpt

```text
0001: {
0002:   "passed": true,
0003:   "errors": [],
0004:   "warnings": [
0005:     "candidates.aggressive.data.voxel_size requires user approval.",
0006:     "candidates.aggressive.model.matching_radius requires user approval.",
0007:     "candidates.aggressive.losses.r_p requires user approval.",
0008:     "candidates.aggressive.losses.r_n requires user approval.",
0009:     "candidates.aggressive.eval.acceptance_radius requires user approval."
0010:   ],
0011:   "candidate_count": 3,
0012:   "validated_candidates": [
0013:     "aggressive",
0014:     "conservative",
0015:     "default"
0016:   ],
0017:   "normalized_params": {
0018:     "conservative": {
0019:       "data.num_points": {
0020:         "value": 8192,
0021:         "status": "benchmark_owned",
0022:         "evidence": [
0023:           {
0024:             "path": "src/benchmarking/hparam_transfer/SKILL.md",
0025:             "field": "Main-Table Point Budget"
0026:           },
0027:           {
0028:             "path": "docs/Benchmark_Efficiency_Table_Design.md",
0029:             "field": "primary main table raw point budget"
0030:           }
0031:         ],
0032:         "transfer_basis": "C3VD \u4e3b\u699c raw point budget \u662f benchmark-owned policy\uff0c\u4e0d\u4ece RegTR vendor config \u8fc1\u79fb\u3002"
0033:       },
0034:       "preprocess.normalize_mode": {
0035:         "value": "none",
0036:         "status": "model_private",
0037:         "evidence": [
0038:           {
0039:             "path": "src/benchmarking/registry/model_registry.py",
0040:             "field": "default_eval_normalize_mode.regtr"
0041:           },
0042:           {
0043:             "path": "src/benchmarking/bridges/configs/c3vd_regtr.yaml",
0044:             "field": "dataset.normalize_mode"
0045:           }
0046:         ],
0047:         "transfer_basis": "RegTR \u7684 3DMatch route \u662f raw metric route\uff1bC3VD train/eval \u5747\u4fdd\u6301 none\u3002"
0048:       },
0049:       "model.private_normalization_route": {
0050:         "value": "bn3_metric_unit",
0051:         "status": "model_private",
0052:         "evidence": [
0053:           {
0054:             "path": "src/benchmarking/registry/model_registry.py",
0055:             "field": "private_input_transform_id.regtr"
0056:           }
0057:         ],
0058:         "transfer_basis": "\u4fdd\u6301 registry \u58f0\u660e\u7684 RegTR metric-unit private route\uff0c\u4e0d\u5f15\u5165 unit_cube/object normalization\u3002"
0059:       },
0060:       "data.voxel_size": {
0061:         "value": 2.0,
0062:         "status": "transferred",
0063:         "evidence": [
0064:           {
0065:             "path": "baselines/RegTR/src/conf/3dmatch.yaml",
0066:             "field": "kpconv_options.first_subsampling_dl"
0067:           },
0068:           {
0069:             "path": "src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json",
0070:             "field": "geometry.nearest_neighbor_spacing"
0071:           },
0072:           {
```
