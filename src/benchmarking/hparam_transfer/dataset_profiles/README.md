# Dataset Profiles

This directory is the canonical repository-local location for common dataset
profiles used by DPG-HPT.

- `dataset_profiles.yaml` is the durable registry for C3VD raycasting and common
  point-cloud registration reference datasets.
- `*.json` and `*.md` files are materialized dataset profile artifacts derived
  from the registry or measured C3VD profile.
- Model-specific route audits do not belong here. They should be written as
  `<model>_reference_profiles.{json,md}` under the relevant hparam-transfer run
  output directory.

