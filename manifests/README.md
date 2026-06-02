# Manifests

`c3vd_raycasting_v1.jsonl` is the stable C3VD raycasting pair manifest used by the portable benchmark configs. Paths inside the manifest are relative to `data.dataset_root`, so users can move the C3VD data root without rewriting every row.

If the raw C3VD-derived point clouds are regenerated, rebuild the manifest with:

```bash
python scripts/benchmark/build_c3vd_raycasting_manifest.py \
  --data-root /path/to/C3VD_sever_datasets \
  --subset-config-out configs/subset_config.json \
  --manifest-out manifests/c3vd_raycasting_v1.jsonl
```
