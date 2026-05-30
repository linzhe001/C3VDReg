# 快速结果概览

## 先看什么
1. `summary_overview.md`: 先看这一页，快速确认本次 run 的关键结论。
2. `leaderboard/leaderboard_main.csv`: 主结果表，优先看 RR@5deg/5mm、边际命中率、RRE、RTE、trimmed Chamfer、训练时间和训练峰值显存。
3. `leaderboard/efficiency_summary.csv`: 速度与资源表，优先看训练时间、训练峰值显存、preprocess/inference/latency。
4. `geometry/geometry_summary.csv`: 几何误差摘要，优先看 visible_nn_mean 和 chamfer。
5. `report.html`: 图表和链接总览。

## 关键指标
- model_id: regtr
- preprocess_profile_id: canonical_v1
- sample_count: 2088
- registration_recall@rre_5deg_rte_5mm: 0.0550766
- rot_hit_5deg_rate: 0.188697
- trans_hit_5mm_rate: 0.0560345
- rre_deg_mean: 32.3068
- rre_deg_median: 19.9452
- rre_deg_p90: 78.9173
- rte_mm_mean: 132.958
- rte_mm_median: 72.6284
- rte_mm_p90: 336.195
- visible_nn_mean_mm_mean: 7.21363
- trimmed_chamfer_mm_mean: 7.2043
- train_time_ms: 4.11168e+07
- train_peak_memory_mb: 1802

## 速度与资源
- train_time_ms: 4.11168e+07
- train_peak_memory_mb: 1802
- train_peak_allocated_mb: 1216.03
- preprocess_time_ms_mean: 4.68461
- inference_time_ms_mean: 68.4392
- refinement_time_ms_mean: 0
- latency_ms_mean: 73.1238
- latency_ms_p90: 99.373
- peak_memory_mb_mean: 0

## 几何摘要
- visible_nn_mean_mm_mean: 7.21363
- trimmed_chamfer_mm_mean: 7.2043
