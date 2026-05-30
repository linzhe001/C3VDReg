# 快速结果概览

## 先看什么
1. `summary_overview.md`: 先看这一页，快速确认本次 run 的关键结论。
2. `leaderboard/leaderboard_main.csv`: 主结果表，优先看 RR@5deg/5mm、边际命中率、RRE、RTE、训练时间和训练峰值显存。
3. `leaderboard/efficiency_summary.csv`: 速度与资源表，优先看训练时间、训练峰值显存、preprocess/inference/latency。
4. `geometry/geometry_summary.csv`: 几何误差摘要，优先看 visible_nn_mean 和 chamfer。
5. `report.html`: 图表和链接总览。

## 关键指标
- model_id: icp
- preprocess_profile_id: canonical_v1
- sample_count: 2088
- registration_recall@rre_5deg_rte_5mm: 0.0545977
- rot_hit_5deg_rate: 0.211207
- trans_hit_5mm_rate: 0.0569923
- rre_deg_mean: 54.6243
- rre_deg_median: 35.5562
- rre_deg_p90: 138.294
- rte_mm_mean: 217.211
- rte_mm_median: 123.721
- rte_mm_p90: 570.218
- visible_nn_mean_mm_mean: 3.1529
- train_time_ms: n/a
- train_peak_memory_mb: n/a

## 速度与资源
- train_time_ms: n/a
- train_peak_memory_mb: n/a
- train_peak_allocated_mb: n/a
- preprocess_time_ms_mean: 4.9707
- inference_time_ms_mean: 782.751
- refinement_time_ms_mean: 0
- latency_ms_mean: 787.722
- latency_ms_p90: 946.329
- peak_memory_mb_mean: 0

## 几何摘要
- visible_nn_mean_mm_mean: 3.1529
- trimmed_chamfer_mm_mean: 3.88742
