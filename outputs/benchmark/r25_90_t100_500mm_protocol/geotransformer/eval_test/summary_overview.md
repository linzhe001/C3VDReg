# 快速结果概览

## 先看什么
1. `summary_overview.md`: 先看这一页，快速确认本次 run 的关键结论。
2. `leaderboard/leaderboard_main.csv`: 主结果表，优先看 RR@5deg/5mm、边际命中率、RRE、RTE、训练时间和训练峰值显存。
3. `leaderboard/efficiency_summary.csv`: 速度与资源表，优先看训练时间、训练峰值显存、preprocess/inference/latency。
4. `geometry/geometry_summary.csv`: 几何误差摘要，优先看 visible_nn_mean 和 chamfer。
5. `report.html`: 图表和链接总览。

## 关键指标
- model_id: geotransformer
- preprocess_profile_id: canonical_v1
- sample_count: 2088
- registration_recall@rre_5deg_rte_5mm: 0.17433
- rot_hit_5deg_rate: 0.561782
- trans_hit_5mm_rate: 0.181513
- rre_deg_mean: 10.7891
- rre_deg_median: 4.28571
- rre_deg_p90: 17.1557
- rte_mm_mean: 44.2163
- rte_mm_median: 15.8558
- rte_mm_p90: 82.1235
- visible_nn_mean_mm_mean: 1.48711
- train_time_ms: 1.54877e+07
- train_peak_memory_mb: 692

## 速度与资源
- train_time_ms: 1.54877e+07
- train_peak_memory_mb: 692
- train_peak_allocated_mb: 496.905
- preprocess_time_ms_mean: 4.80239
- inference_time_ms_mean: 181.96
- refinement_time_ms_mean: 0
- latency_ms_mean: 186.762
- latency_ms_p90: 267.147
- peak_memory_mb_mean: 0

## 几何摘要
- visible_nn_mean_mm_mean: 1.48711
- trimmed_chamfer_mm_mean: 2.42208
