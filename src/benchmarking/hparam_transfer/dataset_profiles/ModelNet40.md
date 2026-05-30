# Dataset Profile: ModelNet40

## Summary

- profile_version: `1`
- profile_source: `baseline_config_reference`
- domain: `normalized_object`
- pair_type: `full_to_full`
- coordinate_unit: `unit_cube`
- inferred_unit: `unit_scale`
- split_policy: `vendor_defined`
- split: `train_test`
- scene_count: `None`
- pair_count_selected: `None`
- pair_count_measured: `None`

## Geometry

| Metric | p10 | p50 | p90 |
| --- | ---: | ---: | ---: |
| source_point_count | None | None | None |
| target_point_count | None | None | None |
| source_bbox_diag | None | 1.0 | None |
| target_bbox_diag | None | 1.0 | None |
| source_nn_spacing | None | None | None |
| target_nn_spacing | None | None | None |

## Pose

- storage_field: `None`
- storage_field_present_fraction: `None`
- pose_source: `unknown`
- transform_format: `unknown`
- transform_shapes: `unknown`
- direction: `unknown`
- direction_confidence: `missing_profile_evidence`
- translation_unit: `unit_cube`
- valid_se3_fraction: `None`
- checked_record_count: `None`
- explicit_transform_count: `None`
- default_identity_count: `None`

| Pose Metric | p10 | p50 | p90 |
| --- | ---: | ---: | ---: |
| translation_norm | None | None | None |
| rotation_determinant | None | None | None |
| rotation_orthonormal_error | None | None | None |

## Digests

- manifest_digest: `None`
- profile_digest: `85d65db4994a640fdd897b6e010deb678a46adbb8868c848d0fe48648f42a669`

## Notes

- Coverage summary: 12311 CAD models across 40 categories, with 9843 train shapes and 2468 test shapes.
- Common point-cloud route summary: baselines often sample 1024 or 2048 points per object; local DCP/PointNetLK reference routes use 1024.
- ModelNet40 is a full-object CAD classification/source route, not a natural partial scene registration dataset.
- In this benchmark it should be treated as normalized-object evidence, not as a default C3VD route.
- Geometry statistics are route-level assumptions from common sampled-point usage and local reference configs, not local ModelNet40 measurements.
