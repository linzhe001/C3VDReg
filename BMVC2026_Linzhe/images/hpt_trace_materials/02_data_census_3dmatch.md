# Data Census: Source 3DMatch

## Sources
- `/home/linzhe/PCLR_compare/src/benchmarking/hparam_transfer/dataset_profiles/3DMatch.json`
- `/home/linzhe/PCLR_compare/baselines/RegTR/src/conf/3dmatch.yaml`

## source reference profile

```text
dataset_id: 3DMatch
domain: indoor_metric_scene
pair_type: partial_to_partial
coordinate_unit: m
inferred_unit: m
split: 46 train / 8 val / 8 test scenes (registry note)
overlap: commonly >30% registration-pair route
```

## registry geometry availability

```text
source_point_count     p10=    None p50=    None p90=    None
target_point_count     p10=    None p50=    None p90=    None
source_bbox_diag       p10=    None p50=    None p90=    None
target_bbox_diag       p10=    None p50=    None p90=    None
source_nn_spacing      p10=    None p50=    None p90=    None
target_nn_spacing      p10=    None p50=    None p90=    None
```

## RegTR 3DMatch scale evidence

```text
dataset.overlap_radius: 0.0375
kpconv_options.first_subsampling_dl: 0.025
losses.r_p / r_n: 0.2 / 0.4
validation.reg_success_thresh_trans: 0.1
```
