# External Baselines

Baseline source trees are not tracked in this repository. The benchmark runtime expects users to clone each upstream or maintained baseline repository into the fixed paths below before running the corresponding model config.

```text
baselines/dcp
baselines/PointNetLK
baselines/PointNetLK_Revisited
baselines/RegTR
baselines/GeoTransformer
baselines/BUFFER-X
```

Use normal `git clone` commands, for example:

```bash
git clone <dcp-repo-url> baselines/dcp
git clone <pointnetlk-repo-url> baselines/PointNetLK
git clone <pointnetlk-revisited-repo-url> baselines/PointNetLK_Revisited
git clone <regtr-repo-url> baselines/RegTR
git clone <geotransformer-repo-url> baselines/GeoTransformer
git clone <bufferx-repo-url> baselines/BUFFER-X
```

The train/eval runners intentionally check that vendor baselines are independent git repositories and have no tracked local modifications. Keep C3VDReg changes in `src/`, `scripts/`, and `configs/`, not inside these external baseline trees.
