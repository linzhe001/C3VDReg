# Model Route Audit: RegTR 3DMatch

## Sources
- `/home/linzhe/PCLR_compare/configs/benchmark/hparam_transfer/baseline_routes.yaml`
- `/home/linzhe/PCLR_compare/baselines/RegTR/src/conf/3dmatch.yaml`
- `/home/linzhe/PCLR_compare/baselines/RegTR/src/train.py`

## route card evidence

```text
model: regtr
known_route: 3DMatch -> indoor_metric_scene
config: baselines/RegTR/src/conf/3dmatch.yaml
preferred_for_c3vd: 3DMatch, 3DLoMatch
reject_for_c3vd_by_default: ModelNet40, KITTI
```

## vendor 3DMatch config excerpt

```text
0001: general:
0002:     expt_name: regtr_regressCoor
0003: 
0004: dataset:
0005:     dataset: 3dmatch
0006:     root: '../data/indoor'
0007:     augment_noise: 0.005
0008:     perturb_pose: small
0009:     train_batch_size: 2
0010:     val_batch_size: 2
0011:     test_batch_size: 1
0012:     overlap_radius: 0.0375  # Distance below which points will be considered to be overlapping
0013: 
0014: train_options:
0015:     niter: -70  # Actually just need 40-50 epochs.
0016: 
0017: solver:
0018:     optimizer: AdamW
0019:     base_lr: 0.0001
0020:     weight_decay: 0.0001
0021:     grad_clip: 0.1
0022:     scheduler: 'step'
0023:     scheduler_param: [205860, 0.5]  # Decay by 0.5 every 20 epochs
0024: 
0025: 
0026: # Use the same processing or backbone as Predator
0027: kpconv_options:
0028:     num_layers: 4
0029:     neighborhood_limits: [40, 40, 40, 40]
0030:     aggregation_mode: sum
0031:     first_subsampling_dl: 0.025
0032:     first_feats_dim: 128
0033:     fixed_kernel_points: center
0034:     in_feats_dim: 1
0035:     in_points_dim: 3
0036:     conv_radius: 2.5
```

## loss/eval scale fields

```text
0088:     # Feature loss - I use the following thresholds
0089:     # Voxel sizes at different octaves: (0) 0.025, (1) 0.05, (2) 0.1, (3) 0.2
0090:     # r_p and r_n are set to 1x and 2.0x the voxel sizes respectively
0091:     wt_feature: 0.1
0092:     wt_feature_un: 0.0
0093:     r_p: 0.2
0094:     r_n: 0.4
0095:     feature_loss_on: [5]
0096:     feature_loss_type: infonce
0097: 
0098:     # Correspondence loss
0099:     wt_corr: 1.0
0100:     corr_loss_on: [5]
0101: 
0102: 
0103: validation:
0104:     # Registration success criteria. We use this to pick the best checkpoint
0105:     reg_success_thresh_rot: 10
0106:     reg_success_thresh_trans: 0.1
```

## training entrypoint excerpt

```text
0001: import os, argparse
0002: 
0003: from easydict import EasyDict
0004: 
0005: from cvhelpers.misc import prepare_logger
0006: from cvhelpers.torch_helpers import setup_seed
0007: 
0008: from data_loaders import get_dataloader
0009: from models import get_model
0010: from trainer import Trainer
0011: from utils.misc import load_config
0012: 
0013: # setup_seed(0, cudnn_deterministic=False)
0014: 
0015: #############
0016: # Argparse. We use command line arguments for training options.
0017: # Model and dataset options are stored in the .yaml config file
0018: #############
0019: parser = argparse.ArgumentParser()
0020: # General
0021: parser.add_argument('--config', type=str, help='Path to the config file.')
0022: # Logging
0023: parser.add_argument('--logdir', type=str, default='../logs',
0024:                     help='Directory to store logs, summaries, checkpoints.')
0025: parser.add_argument('--dev', action='store_true',
0026:                     help='If true, will ignore logdir and log to ../logdev instead')
0027: parser.add_argument('--name', type=str,
0028:                     help='Experiment name (used to name output directory')
0029: parser.add_argument('--summary_every', type=int, default=500,
0030:                     help='Interval to save tensorboard summaries')
0031: parser.add_argument('--validate_every', type=int, default=-1,
0032:                     help='Validation interval. Default: every epoch')
0033: parser.add_argument('--debug', action='store_true',
0034:                     help='If set, will enable autograd anomaly detection')
0035: # Misc
0036: parser.add_argument('--num_workers', type=int, default=4,
0037:                     help='Number of worker threads for dataloader')
0038: # Training and model options
0039: parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
0040: parser.add_argument('--nb_sanity_val_steps', type=int, default=2,
0041:                     help='Number of validation sanity steps to run before training.')
0042: 
0043: opt = parser.parse_args()
0044: # Override config if --resume is passed
0045: if opt.config is None:
0046:     if opt.resume is None or not os.path.exists(opt.resume):
0047:         print('--config needs to be supplied unless resuming from checkpoint')
0048:         exit(-1)
```
