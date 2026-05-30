# 快速结果概览

## 先看什么
1. `summary_overview.md`: 先看这一页，快速确认本次 run 的关键结论。
2. `leaderboard/leaderboard_main.csv`: 主结果表，优先看 RR@5deg/5mm、边际命中率、RRE、RTE、训练时间和训练峰值显存。
3. `leaderboard/efficiency_summary.csv`: 速度与资源表，优先看训练时间、训练峰值显存、preprocess/inference/latency。
4. `geometry/geometry_summary.csv`: 几何误差摘要，优先看 visible_nn_mean 和 chamfer。
5. `report.html`: 图表和链接总览。

## 关键指标
- model_id: pointnetlk_revisited
- preprocess_profile_id: canonical_v1
- sample_count: 2088
- registration_recall@rre_5deg_rte_5mm: 0.0239464
- rot_hit_5deg_rate: 0.303161
- trans_hit_5mm_rate: 0.0277778
- rre_deg_mean: 30.6241
- rre_deg_median: 7.39949
- rre_deg_p90: 106.407
- rte_mm_mean: 119.22
- rte_mm_median: 34.6163
- rte_mm_p90: 374.878
- visible_nn_mean_mm_mean: 2.97088
- train_time_ms: 4.31469e+06
- train_peak_memory_mb: 5180

## 速度与资源
- train_time_ms: 4.31469e+06
- train_peak_memory_mb: 5180
- train_peak_allocated_mb: 3480.44
- preprocess_time_ms_mean: 4.76064
- inference_time_ms_mean: 127.367
- refinement_time_ms_mean: 0
- latency_ms_mean: 132.127
- latency_ms_p90: 162.871
- peak_memory_mb_mean: 0

## 几何摘要
- visible_nn_mean_mm_mean: 2.97088
- trimmed_chamfer_mm_mean: 3.69103
