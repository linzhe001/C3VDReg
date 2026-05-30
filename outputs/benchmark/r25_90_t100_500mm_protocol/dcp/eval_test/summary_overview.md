# 快速结果概览

## 先看什么
1. `summary_overview.md`: 先看这一页，快速确认本次 run 的关键结论。
2. `leaderboard/leaderboard_main.csv`: 主结果表，优先看 RR@5deg/5mm、边际命中率、RRE、RTE、训练时间和训练峰值显存。
3. `leaderboard/efficiency_summary.csv`: 速度与资源表，优先看训练时间、训练峰值显存、preprocess/inference/latency。
4. `geometry/geometry_summary.csv`: 几何误差摘要，优先看 visible_nn_mean 和 chamfer。
5. `report.html`: 图表和链接总览。

## 关键指标
- model_id: dcp
- preprocess_profile_id: canonical_v1
- sample_count: 2088
- registration_recall@rre_5deg_rte_5mm: 0
- rot_hit_5deg_rate: 0
- trans_hit_5mm_rate: 0
- rre_deg_mean: 84.1516
- rre_deg_median: 72.9487
- rre_deg_p90: 155.145
- rte_mm_mean: 313.215
- rte_mm_median: 279.078
- rte_mm_p90: 565.078
- visible_nn_mean_mm_mean: 13.5146
- train_time_ms: 5.82757e+07
- train_peak_memory_mb: 12600

## 速度与资源
- train_time_ms: 5.82757e+07
- train_peak_memory_mb: 12600
- train_peak_allocated_mb: 12077.7
- preprocess_time_ms_mean: 4.84353
- inference_time_ms_mean: 59.4247
- refinement_time_ms_mean: 0
- latency_ms_mean: 64.2683
- latency_ms_p90: 67.3377
- peak_memory_mb_mean: 0

## 几何摘要
- visible_nn_mean_mm_mean: 13.5146
- trimmed_chamfer_mm_mean: 18.1651
