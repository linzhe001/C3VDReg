# Data Census: Target C3VD

## Sources
- `/home/linzhe/PCLR_compare/src/benchmarking/hparam_transfer/dataset_profiles/c3vd_raycasting_v1.json`

## target profile summary

```text
dataset_id: c3vd_raycasting_v1
domain: medical_endoscopic_cross_modal
pair_type: partial_to_partial
coordinate_unit: mm_like
inferred_unit: mm_like
split_policy: scene_safe
split/full_test_pairs: test / 2088
```

## measured geometry

```text
source_point_count     p10=1.925e+04 p50=3.247e+04 p90=4.339e+04
target_point_count     p10=    9650 p50=1.457e+04 p90=3.141e+04
source_bbox_diag       p10=   81.36 p50=   107.1 p90=   116.2
target_bbox_diag       p10=   92.43 p50=     122 p90=   134.9
source_nn_spacing      p10=   0.636 p50=  0.8513 p90=   1.008
target_nn_spacing      p10=  0.7037 p50=  0.8697 p90=   1.215
```

## pose contract

```text
storage_field: gt_transform
pose_source: manifest_default_identity_for_one_to_one
transform_format: homogeneous_SE3
transform_shapes: ['4x4_homogeneous']
direction: source_to_target
translation_unit: mm_like
valid_se3_fraction: 1.0
```
