# Checkpoint Distribution

Recommended policy: do not commit checkpoints to normal git. For public release, use a versioned artifact host with stable URLs and checksums, preferably Zenodo or Hugging Face Hub. Google Drive is acceptable as a temporary private mirror, but it is weaker for paper reproducibility because links, permissions, and download behavior are less stable.

Keep the repository contract simple:

- `SELECTED_CHECKPOINTS.csv` records the expected local path, size, and SHA256.
- `checkpoints/` is a local cache ignored by git.
- Eval configs in `configs/benchmark/paper_r25_90_t100_500mm/` point to `checkpoints/<model>/...`.
- If using Google Drive temporarily, upload one versioned archive such as `c3vdreg_checkpoints_r25_90_t100_500mm_v1.zip` and include this CSV inside the archive.

After downloading, verify:

```bash
sha256sum \
  checkpoints/dcp/model_best.pth \
  checkpoints/geotransformer/geotransformer_c3vd_model_best.pth \
  checkpoints/mamba3d_mamba2_direct/mamba3d_pointlk_model_best.pth \
  checkpoints/pointnetlk/c3vd_pointnetlk_model_model_best.pth \
  checkpoints/pointnetlk_revisited/pointnetlk_c3vd_model_best.pth \
  checkpoints/regtr/model-142848.pth
```
