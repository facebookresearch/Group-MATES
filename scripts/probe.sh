# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH --array=0-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

export CKPT="/data/datasets/hf_cache/dclm_logs/baseline_01_01_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480/checkpoints/epoch_2"

CMD="-m group-mates.oracle.probe_data_influence -- \
  --scale 411m_4x \
  --data-config exp_data/datasets/tokenized/baseline_01_01_fasttext.json \
  --logs /data/datasets/hf_cache/dclm_logs"

SEED=$SLURM_ARRAY_TASK_ID TMPDIR=/scratch/$(whoami)/tmp torchrun --nproc-per-node 2 --master_port $(expr $RANDOM + 1000) $CMD
