# Final Benchmark Config

## Sources
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/promoted/promoted_default.yaml`
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/configs/regtr_eval_full_8192_best_perturbed.yaml`
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/validated/candidate_configs.yaml`

## final benchmark config summary

```text
promoted candidate: default
benchmark: c3vd_raycasting_v1 | point_unit=mm_like
model: regtr
normalize_mode: none
num_points_override: 8192 in final eval config
overrides: voxel=2.5, radius=3.75, r_p=20.0, r_n=40.0,
  checkpoint-selection translation threshold=10.0
official RR/RTE thresholds remain benchmark-owned.
```

## promoted_default excerpt

```text
0001: benchmark:
0002:   name: c3vd_raycasting_v1
0003:   split: train
0004:   subset_name: null
0005:   point_unit: mm_like
0006:   official_track: main
0007: data:
0008:   manifest_path: /home/linzhe/PCLR_compare/outputs/benchmark/flow_check_eval/c3vd_raycasting_manifest.jsonl
0009:   subset_config_path: /home/linzhe/PCLR_compare/configs/subset_config.json
0010:   dataset_root: /mnt/f/Datasets/C3VD_sever_datasets
0011: preprocess:
0012:   profile: canonical_v1
0013:   seed: 42
0014:   sampling_override: null
0015:   num_points_override: null
0016: model:
0017:   id: regtr
0018:   checkpoint_path: null
0019:   overrides:
0020:     normalize_mode: none
0021:     d_embed: 256
0022:     nhead: 8
0023:     d_feedforward: 1024
0024:     num_encoder_layers: 6
0025:     data.voxel_size: 2.5
0026:     model.matching_radius: 3.75
0027:     losses.r_p: 20.0
0028:     losses.r_n: 40.0
0029:     eval.acceptance_radius: 10.0
0030: perturbation:
0031:   enabled: true
0032:   rotation_deg: 45.0
0033:   translation_m: 0.5
0034:   noise_sigma: 0.0
0035:   noise_clip: 0.0
0036:   apply_noise_to: source
0037: runtime:
0038:   device: cuda:0
0039:   batch_size: 1
0040:   num_workers: 4
0041:   export_html: false
0042:   output_dir: outputs/benchmark/hparam_transfer/regtr_measured_run/train_default
0043:   train_mode: full
0044:   train_metadata_path: null
0045:   training_overrides: {}
0046: analysis:
0047:   required_tables:
0048:   - leaderboard_main
0049:   - leaderboard_multithreshold
0050:   - bucket_overlap
0051:   - bucket_rotation
0052:   - bucket_scene
0053:   - efficiency_summary
0054:   - geometry_summary
0055:   required_curves:
0056:   - rr_multithreshold
0057:   - success_latency_pareto
0058:   bucket_keys:
0059:   - overlap_bin
0060:   - rotation_bin
0061:   - translation_bin
0062:   - scene_id
0063:   - preprocess_profile_id
0064:   - refinement_track
0065:   geometry:
0066:     sample_count: 2048
0067:     distance_mode: visible_overlap_preferred
0068:     export_histogram: true
0069:     export_cdf: true
0070:   qualitative:
0071:     topk_failures: 20
0072:     export_failure_gallery: true
0073:   export:
0074:     html: false
0075:     png: true
0076:     markdown_tables: true
```

## final eval config excerpt

```text
0001: benchmark:
0002:   name: c3vd_raycasting_v1
0003:   split: test
0004:   subset_name: null
0005:   point_unit: mm_like
0006:   official_track: main
0007: 
0008: data:
0009:   manifest_path: /home/linzhe/PCLR_compare/outputs/benchmark/flow_check_eval/c3vd_raycasting_manifest.jsonl
0010:   subset_config_path: /home/linzhe/PCLR_compare/configs/subset_config.json
0011:   dataset_root: /mnt/f/Datasets/C3VD_sever_datasets
0012: 
0013: preprocess:
0014:   profile: canonical_v1
0015:   seed: 42
0016:   sampling_override: null
0017:   num_points_override: 8192
0018: 
0019: perturbation:
0020:   enabled: true
0021:   rotation_deg: 45.0
0022:   translation_m: 0.5
0023:   noise_sigma: 0.0
0024:   noise_clip: 0.0
0025:   apply_noise_to: source
0026: 
0027: model:
0028:   id: regtr
0029:   checkpoint_path: /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_logs/c3vd/260430_233010_benchmark_full_regtr_resume_2976_to_50e/ckpt/model-122016.pth
0030:   overrides:
0031:     normalize_mode: none
0032:     d_embed: 256
0033:     nhead: 8
0034:     d_feedforward: 1024
0035:     num_encoder_layers: 6
0036:     first_subsampling_dl: 2.5
0037:     overlap_radius: 3.75
0038:     r_p: 20.0
0039:     r_n: 40.0
0040:     reg_success_thresh_trans: 10.0
0041: 
0042: runtime:
0043:   device: cuda:0
0044:   batch_size: 1
0045:   num_workers: 0
0046:   export_html: false
0047:   output_dir: /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/eval_full_8192_best_perturbed
0048:   train_metadata_path: /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_metrics/regtr_train_resume_2976_to_50e.json
0049: 
0050: analysis:
0051:   required_tables:
0052:     - leaderboard_main
0053:     - leaderboard_multithreshold
0054:     - bucket_overlap
0055:     - bucket_rotation
0056:     - bucket_scene
0057:     - efficiency_summary
0058:     - geometry_summary
0059:   required_curves:
0060:     - rr_multithreshold
0061:     - success_latency_pareto
0062:   bucket_keys:
0063:     - overlap_bin
0064:     - rotation_bin
0065:     - translation_bin
0066:     - scene_id
0067:     - preprocess_profile_id
0068:     - refinement_track
0069:   geometry:
0070:     sample_count: 2048
0071:     distance_mode: visible_overlap_preferred
0072:     export_histogram: true
0073:     export_cdf: true
0074:   qualitative:
0075:     topk_failures: 20
0076:     export_failure_gallery: false
0077:   export:
0078:     html: false
0079:     png: true
0080:     markdown_tables: true
```
