# Smoke Sanity

## Sources
- `/home/linzhe/PCLR_compare/src/benchmarking/hparam_transfer/candidate_validation.py`
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/candidate_validation/validation_summary.json`
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_metrics/regtr_train.json`
- `/home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_logs/c3vd/260430_210548_benchmark_full_regtr/log.txt`

## smoke sanity pseudo-code

```text
bundle = load_candidate_bundle(candidate_configs.yaml)
runtime = load_benchmark_config(runtime_config.yaml)
for each candidate:
  deep-merge candidate overrides into runtime config
  set output_dir and hparam_transfer_candidate metadata
  if execute_eval:
    run_eval(config)
    reject crash/OOM/missing metrics/all-fail collapse
  else:
    record dry_run with built output path
```

## candidate-validation output

```text
execute_eval: False
conservative/default/aggressive: dry_run (execute_eval=false)
passed_candidates: []
failed_candidates: ['conservative', 'default', 'aggressive']
This retained DPG-HPT gate is config-build/dry-run evidence,
not official C3VD test-score selection.
```

## RegTR train smoke task output

```text
script: src/benchmarking/bridges/train_regtr_c3vd.py
config: regtr_smoke_bridge_config.yaml
task: create loaders/model, run 1 validation sanity step
wrapper_return_code: 0
torch_peak_allocated_mb: 1213.6259765625
command and output evidence from log:
0001: 04/30 21:05:48 [INFO] root - Output and logs will be saved to /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_logs/c3vd/260430_210548_benchmark_full_regtr
0002: 04/30 21:05:48 [INFO] cvhelpers.misc - Command: /home/linzhe/PCLR_compare/src/benchmarking/bridges/train_regtr_c3vd.py --config /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/regtr_smoke_bridge_config.yaml --logdir /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_logs --name benchmark_full_regtr --validate_every -1 --num_workers 4 --nb_sanity_val_steps 1
0003: 04/30 21:05:48 [INFO] cvhelpers.misc - Source is from Commit 11c3afdc (2026-04-30): train(baseline/regtr): align C3VD pose direction with target space
0004: 04/30 21:05:48 [INFO] cvhelpers.misc - Arguments: config: /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/regtr_smoke_bridge_config.yaml, logdir: /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_logs/c3vd, dev: False, name: benchmark_full_regtr, summary_every: 500, validate_every: -1, debug: False, num_workers: 4, data_root: None, resume: None, nb_sanity_val_steps: 1
0005: 04/30 21:05:48 [INFO] root - Logging to: /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_logs/c3vd/260430_210548_benchmark_full_regtr
0006: 04/30 21:05:48 [INFO] root - Config saved to: /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_logs/c3vd/260430_210548_benchmark_full_regtr/config.yaml
0007: 04/30 21:05:48 [INFO] root - Random seed set to: 1234
0008: 04/30 21:05:48 [INFO] root - 
0009: ============================================================
0010: 04/30 21:05:48 [INFO] root - Creating data loaders...
0011: 04/30 21:05:48 [INFO] root - ============================================================
0012: 04/30 21:06:09 [INFO] root - Train batches: 2976
0013: 04/30 21:06:09 [INFO] root - Val batches: 988
0014: 04/30 21:06:09 [INFO] root - 
0015: ============================================================
0016: 04/30 21:06:09 [INFO] root - Creating model...
0017: 04/30 21:06:09 [INFO] root - ============================================================
0018: 04/30 21:06:09 [INFO] RegTR - Instantiating model RegTR
0019: 04/30 21:06:09 [INFO] RegTR - Loss weighting: {'overlap_5': 1.0, 'feature_5': 0.1, 'corr_5': 1.0, 'feature_un': 0.0}
0020: 04/30 21:06:09 [INFO] RegTR - Config: d_embed:256, nheads:8, pre_norm:True, use_pos_emb:True, sa_val_has_pos_emb:True, ca_val_has_pos_emb:True
0021: 04/30 21:06:09 [INFO] root - Total parameters: 11,845,811
0022: 04/30 21:06:09 [INFO] root - Trainable parameters: 11,845,316
0042: 04/30 21:06:10 [INFO] trainer - Validation interval set to 2976 steps
0043: 04/30 21:06:10 [INFO] trainer - Performing validation dry run with 1 steps
0044: 04/30 21:06:11 [INFO] RegTR - Aggregating metrics, total number of instances: 2
0045: 04/30 21:06:11 [INFO] trainer - Validation ended:
0046: [Losses] corr_5: 244.9, feature_5: 15.24, overlap_5: 1.027, total: 247.4
0047: [Metrics] reg_success_0: 0, reg_success_1: 0, reg_success_2: 0, reg_success_3: 0, reg_success_4: 0, reg_success_final: 0, rot_err_deg_0: 178.1, rot_err_deg_1: 178.2, rot_err_deg_2: 178.2, rot_err_deg_3: 178.6, rot_err_deg_4: 177.8, rot_err_deg_final: 177.3, trans_err_0: 81.45, trans_err_1: 82.05, trans_err_2: 82.33, trans_err_3: 82.47, trans_err_4: 82.55, trans_err_final: 82.6
0048: 04/30 21:06:11 [INFO] CheckPointManager - Saved checkpoint: /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_logs/c3vd/260430_210548_benchmark_full_regtr/ckpt/model-0.pth
0049: 04/30 21:06:11 [INFO] CheckPointManager - Checkpoint is current best, score=0.0
0050: 04/30 21:06:11 [INFO] trainer - Starting epoch 0 (steps 0 - 2976)
0051: 04/30 21:16:02 [INFO] trainer - Running validation (step 2976)...
0052: 04/30 21:17:58 [INFO] RegTR - Aggregating metrics, total number of instances: 1975
0053: 04/30 21:17:58 [INFO] trainer - Validation ended:
0054: [Losses] corr_5: 33.46, feature_5: 1.534, overlap_5: 0.8357, total: 34.45
0055: [Metrics] reg_success_0: 0.01266, reg_success_1: 0.05114, reg_success_2: 0.1828, reg_success_3: 0.3008, reg_success_4: 0.36, reg_success_final: 0.3706, rot_err_deg_0: 111.1, rot_err_deg_1: 85.87, rot_err_deg_2: 65.44, rot_err_deg_3: 41.64, rot_err_deg_4: 32.87, rot_err_deg_final: 32.21, trans_err_0: 149.3, trans_err_1: 141.6, trans_err_2: 122.1, trans_err_3: 73.83, trans_err_4: 47.34, trans_err_final: 42.32
0056: 04/30 21:17:58 [INFO] CheckPointManager - Saved checkpoint: /home/linzhe/PCLR_compare/outputs/benchmark/hparam_transfer/regtr_measured_run/train_default/train_bridge_logs/c3vd/260430_210548_benchmark_full_regtr/ckpt/model-2976.pth
0057: 04/30 21:17:58 [INFO] CheckPointManager - Checkpoint is current best, score=0.37063291668891907
0058: 04/30 21:17:58 [INFO] trainer - Epoch 0 complete in 12m17s. Average train losses: corr_5: 42.06, feature_5: 0.6339, overlap_5: 0.6251, total: 42.75
```
