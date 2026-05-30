# Dataset Profile: c3vd_raycasting_v1

## Summary

- profile_version: `1`
- profile_source: `measured`
- domain: `medical_endoscopic_cross_modal`
- pair_type: `partial_to_partial`
- coordinate_unit: `mm_like`
- inferred_unit: `mm_like`
- split_policy: `scene_safe`
- split: `test`
- scene_count: `6`
- pair_count_selected: `212`
- pair_count_measured: `18`

## Geometry

| Metric | p10 | p50 | p90 |
| --- | ---: | ---: | ---: |
| source_point_count | 19254.4 | 32472.5 | 43386.9 |
| target_point_count | 9649.9 | 14572.5 | 31406.3 |
| source_bbox_diag | 81.36053848266602 | 107.09344100952148 | 116.20752868652343 |
| target_bbox_diag | 92.43143310546876 | 121.99967193603516 | 134.89095611572264 |
| source_nn_spacing | 0.6360141396522522 | 0.8513259589672089 | 1.0084967374801637 |
| target_nn_spacing | 0.7036534667015075 | 0.869653195142746 | 1.2152980804443358 |

## Pose

- storage_field: `gt_transform`
- storage_field_present_fraction: `0.0`
- pose_source: `manifest_default_identity_for_one_to_one`
- transform_format: `homogeneous_SE3`
- transform_shapes: `4x4_homogeneous`
- direction: `source_to_target`
- direction_confidence: `benchmark_manifest_contract`
- translation_unit: `mm_like`
- valid_se3_fraction: `1.0`
- checked_record_count: `18`
- explicit_transform_count: `0`
- default_identity_count: `18`

| Pose Metric | p10 | p50 | p90 |
| --- | ---: | ---: | ---: |
| translation_norm | 0.0 | 0.0 | 0.0 |
| rotation_determinant | 1.0 | 1.0 | 1.0 |
| rotation_orthonormal_error | 0.0 | 0.0 | 0.0 |

## Digests

- manifest_digest: `7cb917972f4fa0ff867e343e344b731be5f8cdb1ae7e8f859f0d03e46cbb5513`
- profile_digest: `d9f5d87abbfed0c81085dad06c765bef398f2d7c7a28323896edbf68f569f4e9`

## Notes

- declared_point_unit=mm_like
- geometry_consistent_with_declared_unit
- pose_storage_field=gt_transform
- pose_source=manifest_default_identity_for_one_to_one
- pose_transform_shapes=4x4_homogeneous
- pose_direction=source_to_target
