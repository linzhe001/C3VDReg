# Dataset Profile: 3DMatch

## Summary

- profile_version: `1`
- profile_source: `baseline_config_reference`
- domain: `indoor_metric_scene`
- pair_type: `partial_to_partial`
- coordinate_unit: `m`
- inferred_unit: `m`
- split_policy: `vendor_defined_scene_split`
- split: `train_val_test`
- scene_count: `62`
- pair_count_selected: `None`
- pair_count_measured: `None`

## Geometry

| Metric | p10 | p50 | p90 |
| --- | ---: | ---: | ---: |
| source_point_count | None | None | None |
| target_point_count | None | None | None |
| source_bbox_diag | None | None | None |
| target_bbox_diag | None | None | None |
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
- translation_unit: `m`
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
- profile_digest: `9d642412034c5fc2a67ef4e39d706ef74a53a5a71435b0b169b6badef922af38`

## Notes

- Coverage summary: 62 scenes split into 46 train, 8 validation, and 8 test scenes.
- Standard 3DMatch registration pairs are commonly treated as overlap greater than 30%.
- 3DMatch is an indoor RGB-D reconstructed fragment registration route, not an object-level normalized route.
- Keep metric-space assumptions unless vendor code explicitly applies private normalization.
- Geometry statistics are registry-level reference facts, not measured from local 3DMatch files in this repo.
