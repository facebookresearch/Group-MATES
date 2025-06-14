# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse

from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer
from file_utils import list_dir, read_jsonl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/data/datasets/hf_cache")
    parser.add_argument("--base", type=int, default=0)
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--map_batch_size", type=int, default=1024)
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)

    args = parser.parse_args()
    print(args)

    data_dir = f"{args.base_dir}/refinedweb_01_0/fasttext/fasttext_filter/processed_data"
    file_list = list_dir(data_dir)
    shard_size = len(file_list) // args.shard[1]
    file_list = file_list[
        args.base
        + args.shard[0]
        * shard_size : (
            args.base + (args.shard[0] + 1) * shard_size
            if args.shard[0] + 1 < args.shard[1]
            else len(file_list)
        )
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased",
        max_length=2048,
        padding="max_length",
    )

    for file in tqdm(file_list):
        if os.path.isdir(file):
            print(file)
            continue

        shard_name = file.split("/")[-1].split(".")[0]
        output_dir = f"{args.base_dir}/refinedweb_01_0/fasttext/fasttext_filter/processed_data/bert_tokenized/{shard_name}"

        dataset = []
        for json_line in read_jsonl(file):
            dataset.append(" ".join(json_line["text"].strip().splitlines()))
        dataset = Dataset.from_list([{"text": d} for d in dataset])
        print("Total number of examples:", len(dataset))

        def preprocess_data(examples):
            texts = examples["text"]
            encoding = tokenizer.batch_encode_plus(
                texts,
                max_length=2048,
                padding="max_length",
                truncation=True,
            )
            return encoding

        dataset = dataset.map(
            preprocess_data,
            batched=True,
            batch_size=args.map_batch_size,
            num_proc=8,
            remove_columns=dataset.column_names,
        )
        print("After tokenization: Total number of examples:", len(dataset))

        dataset.save_to_disk(output_dir)
