# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cargo run --release -- \
    --input /data/datasets/hf_cache/refinedweb_01_0/fasttext/fasttext_filter/processed_data \
    --local-cell-dir /scratch/$(whoami)/tmp \
    --output /data/datasets/hf_cache/baseline_01_0_fasttext_tokenized \
    --tokenizer "EleutherAI/gpt-neox-20b" \
    --seqlen 2049 \
    --wds-chunk-size 8192 \
    --num-local-cells 512
