# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=fasttext
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00

BASE_DIR=/data/datasets/hf_cache
SPILL_LOCATION=/scratch/$(whoami)/tmp/ray
mkdir -p $SPILL_LOCATION
ray start --head --port=6379 --temp-dir=$SPILL_LOCATION

TMPDIR=/scratch/$(whoami)/tmp PYTHONPATH=$(pwd) python ray_processing/process.py \
    --source_ref_paths exp_data/datasets/raw_sources/refinedweb_01_0.json \
    --readable_name fasttext_01_0 \
    --output_dir $BASE_DIR/refinedweb_01_0/fasttext \
    --config_path baselines/baselines_configs/fasttext_filter.yaml \
    --source_name cc \
    --overwrite

ray stop
rm -rf $SPILL_LOCATION
