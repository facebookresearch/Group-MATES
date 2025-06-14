# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 mlfoundations
#
# This code is based on https://github.com/mlfoundations/dclm

import yaml


def update_args_from_openlm_config(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        if k == "model" and args.model != None:
            continue
        if v == "None":
            v = None

        # we changed args
        if k == "batch_size":
            k = "per_gpu_batch_size"
        if k == "val_batch_size":
            k = "per_gpu_val_batch_size"
        if k == "val_data" and args.val_data != None:
            continue

        # For forcing xformers
        if args.force_xformers:
            if k == "attn_name":
                v = "xformers_attn"
            if k == "torchcompile":
                v = False

        setattr(args, k, v)
