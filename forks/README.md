# External Forks

`forks/PointNetLK_c3vd/` is not tracked in this repository. It is required only for the PointNetLK-C3VD/Mamba routes such as `mamba3d`, `mamba3d_mamba2`, `mamba3d_mamba2_direct`, and `mambanetlk`.

Restore it by cloning the maintained PointNetLK-C3VD fork into the expected path:

```bash
git clone <pointnetlk-c3vd-fork-url> forks/PointNetLK_c3vd
```

The adapter imports `ptlk` modules directly from that path, so the directory name must stay unchanged.
