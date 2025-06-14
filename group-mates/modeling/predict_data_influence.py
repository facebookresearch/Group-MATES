# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fsspec
import os

import torch
import datasets
from transformers import AutoTokenizer
from modeling_data_influence_model import RelationalModel

model_dir = f"{os.environ.get('CKPT')}/dim-final"
model = RelationalModel.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
model.to(device)

print(model.temp, model.alpha)

pythia_tokenizer = AutoTokenizer.from_pretrained(
    "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
)
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    max_length=2048,
    padding="max_length",
)


@torch.no_grad()
def embed_batch(example):
    for key, value in example.items():
        bs = len(value)
        example[key] = torch.tensor(value, device="cuda").reshape(bs * 4, -1)
    outputs = model.model(
        example["input_ids"],
        attention_mask=example["attention_mask"],
        token_type_ids=example["token_type_ids"],
        return_dict=True,
    )
    p_reps = outputs.last_hidden_state[:, 0]
    p_reps = torch.nn.functional.normalize(p_reps, dim=-1).contiguous()
    p_reps = p_reps.reshape(-1, 4, p_reps.size(1)).mean(dim=1)
    pooled_output_p = model.dropout(p_reps)
    scores = model.classifier(pooled_output_p).squeeze()
    return p_reps.detach().float().cpu().numpy(), scores.detach().float().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="/data/datasets/hf_cache")
    parser.add_argument("-S", "--shard", type=int, nargs=2, default=[0, 1])
    parser.add_argument("-b", "--device_batch_size", type=int, default=128)

    args = parser.parse_args()
    print(args)

    data_dir = f"{args.base_dir}/refinedweb_01_0/fasttext/fasttext_filter/processed_data/bert_tokenized"
    output_dir = f"{os.environ.get('CKPT')}/dim-prediction"

    fs = fsspec.filesystem("local")
    file_list = fs.glob(data_dir + "/*")
    shard_names = [file.split("/")[-1].split("_bert")[0] for file in file_list]
    shard_size = len(file_list) // args.shard[1]
    print(
        args.shard[0] * shard_size,
        (
            (args.shard[0] + 1) * shard_size
            if args.shard[0] + 1 < args.shard[1]
            else len(file_list)
        ),
    )
    dataset = datasets.concatenate_datasets(
        [
            # datasets.load_from_disk("gs://" + file_list[i])
            datasets.load_from_disk(file_list[i])
            for i in range(
                args.shard[0] * shard_size,
                (
                    (args.shard[0] + 1) * shard_size
                    if args.shard[0] + 1 < args.shard[1]
                    else len(file_list)
                ),
            )
        ]
    )

    print("Before annotation: Total number of examples:", len(dataset))

    dataset = dataset.map(
        lambda x: (lambda r, p: {"reps": r, "prediction": p})(*embed_batch(x)),
        batched=True,
        batch_size=args.device_batch_size,
        remove_columns=dataset.column_names,
    )
    print("After annotation: Total number of examples:", len(dataset))

    print(f"Saving to {output_dir}")
    dataset.save_to_disk(output_dir + f"/{args.shard[0]}")
