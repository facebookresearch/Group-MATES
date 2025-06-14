# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

export HF_HOME=/data/datasets/hf_cache

method="baseline_01_01_fasttext-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=4-seed=124-tokens=32929300480"

PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 torchrun --nproc_per_node 8 --master_port 47762 eval/eval_openlm_ckpt.py \
    --donot-compute-perplexity \
    --checkpoint /data/datasets/hf_cache/dclm_logs/$method/checkpoints/epoch_6.pt \
    --model ../training/open_lm_configs/d=1024_l=24_h=8.json \
    --config /data/datasets/hf_cache/dclm_logs/$method/params.txt \
    --eval-yaml eval/mmlu_and_lowvar.yaml \
    --output-file results/$method/epoch_6/metrics_mmlu_and_lowvar.json \
    --use-temp-working-dir
