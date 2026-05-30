# Dataset Profile: 3DLoMatch

## Summary

- profile_version: `1`
- profile_source: `baseline_config_reference`
- domain: `indoor_metric_scene`
- pair_type: `partial_to_partial`
- coordinate_unit: `m`
- inferred_unit: `m`
- split_policy: `vendor_defined_low_overlap`
- split: `test`
- scene_count: `8`
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
- profile_digest: `1f01e300de893d13e493efd6d1a47375e762aafd38e8c09fa0b9c945c90877e4`

## Notes

- Coverage summary: derived from the 8 3DMatch test scenes.
- Low-overlap route summary: commonly treated as 10%-30% overlap pairs.
- 3DLoMatch shares the 3DMatch metric indoor scene family but emphasizes low-overlap registration.
- Use as robustness evidence only when the baseline exposes a compatible 3DMatch/3DLoMatch route.
