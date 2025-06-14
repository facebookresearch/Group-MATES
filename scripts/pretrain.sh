# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

# print commands
set -x

export WANDB_DIR="/scratch/$(whoami)/tmp"
mkdir -p $WANDB_DIR

torchrun --nproc-per-node 8 -m training.train -- \
  --scale 411m_4x \
  --data-config exp_data/datasets/tokenized/baseline_01_01_fasttext.json \
  --logs /data/datasets/hf_cache/dclm_logs \
  --multiple-data-passes \
  --report-to-wandb
