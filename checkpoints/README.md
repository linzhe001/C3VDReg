# Checkpoints

Model weights are intentionally not tracked in git. Download the checkpoint bundle listed in `CHECKPOINTS.md`, verify SHA256 values against `SELECTED_CHECKPOINTS.csv`, and place the files under this directory.

Expected local layout:

```text
checkpoints/dcp/model_best.pth
checkpoints/geotransformer/geotransformer_c3vd_model_best.pth
checkpoints/mamba3d_mamba2_direct/mamba3d_pointlk_model_best.pth
checkpoints/pointnetlk/c3vd_pointnetlk_model_model_best.pth
checkpoints/pointnetlk_revisited/pointnetlk_c3vd_model_best.pth
checkpoints/regtr/model-142848.pth
```
