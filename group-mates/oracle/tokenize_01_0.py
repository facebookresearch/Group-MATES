# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import webdataset as wds

input_dir = "/data/datasets/hf_cache/baseline_01_0_fasttext_tokenized"
output_dir = "group-mates/oracle/data"


def test_tokenize_shuffle_simple_do_sample():
    dss = [
        wds.WebDataset(os.path.join(input_dir, f"shard_{i:08d}.tar")).decode()
        for i in range(10)
    ]
    data = [torch.tensor([x["json.gz"]], dtype=torch.int32) for ds in dss for x in ds]
    print(len(data))
    torch.save(data, os.path.join(output_dir, "train.pt"))


test_tokenize_shuffle_simple_do_sample()
