#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."

MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp /home/linzhe/anaconda3/envs/PCLR_compare/bin/python scripts/runners/eval_benchmark.py --config outputs/benchmark/r25_90_t100_500mm_protocol/configs/eval_geotransformer.yaml
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp /home/linzhe/anaconda3/envs/PCLR_compare/bin/python scripts/runners/eval_benchmark.py --config outputs/benchmark/r25_90_t100_500mm_protocol/configs/eval_regtr.yaml
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp /home/linzhe/anaconda3/envs/PCLR_compare/bin/python scripts/runners/eval_benchmark.py --config outputs/benchmark/r25_90_t100_500mm_protocol/configs/eval_mamba2_direct.yaml
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp /home/linzhe/anaconda3/envs/PCLR_compare/bin/python scripts/runners/eval_benchmark.py --config outputs/benchmark/r25_90_t100_500mm_protocol/configs/eval_pointnetlk_revisited.yaml
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp /home/linzhe/anaconda3/envs/PCLR_compare/bin/python scripts/runners/eval_benchmark.py --config outputs/benchmark/r25_90_t100_500mm_protocol/configs/eval_pointnetlk.yaml
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp /home/linzhe/anaconda3/envs/PCLR_compare/bin/python scripts/runners/eval_benchmark.py --config outputs/benchmark/r25_90_t100_500mm_protocol/configs/eval_dcp.yaml
MPLCONFIGDIR=/tmp/mplconfig XDG_CACHE_HOME=/tmp /home/linzhe/anaconda3/envs/PCLR_compare/bin/python scripts/runners/eval_benchmark.py --config outputs/benchmark/r25_90_t100_500mm_protocol/configs/eval_icp.yaml
