# R25-90/T100-500mm Hard Protocol Summary

All rows use 2088 C3VD raycasting test pairs, 8192 input points, source-only rotation sampled from 25-90 degrees, translation sampled from 100-500 mm, and no added noise.

| Model | RR@5 | RR@10 | R<=5 | T<=5 | RRE mean | RTE mean | RTE median | RTE p90 | Trim CD | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GeoTransformer | 17.43 | 35.63 | 56.18 | 18.15 | 10.79 | 44.22 | 15.86 | 82.12 | 2.42 | 186.8 |
| ICP | 5.46 | 10.73 | 21.12 | 5.70 | 54.62 | 217.21 | 123.72 | 570.22 | 3.89 | 787.7 |
| RegTR | 5.32 | 11.59 | 18.87 | 5.41 | 32.33 | 133.01 | 72.81 | 335.76 | 6.16 | 83.0 |
| PointNetLK-Mamba | 3.16 | 11.11 | 24.28 | 3.64 | 18.17 | 74.15 | 39.47 | 174.23 | 3.52 | 121.7 |
| PointNetLK Revisited | 2.39 | 10.58 | 30.32 | 2.78 | 30.62 | 119.22 | 34.62 | 374.88 | 3.69 | 132.1 |
| PointNetLK | 0.14 | 1.01 | 2.73 | 0.19 | 47.07 | 194.70 | 154.89 | 413.49 | 5.82 | 93.1 |
| DCP | 0.00 | 0.00 | 0.00 | 0.00 | 84.15 | 313.21 | 279.08 | 565.08 | 18.17 | 64.3 |

Interpretation: `R<=5` and `T<=5` are marginal hit rates. `RR@5` is their joint success and is therefore much lower whenever one error family dominates.
