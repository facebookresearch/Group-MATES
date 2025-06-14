# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 mlfoundations
#
# This code is based on https://github.com/mlfoundations/dclm

import glob
import json
import os
import pathlib
import uuid
from datetime import datetime

import boto3
import git
import yaml
from cloudpathlib import S3Path

DATASET_REFS_DIR = os.path.join(
    pathlib.Path(__file__).parent.parent.absolute(),
    "exp_data",
    "datasets",
)


def get_source_ref_by_key(search_value, key="name", tokenized=False):
    for f in glob.glob(f"{DATASET_REFS_DIR}/*/*.json"):
        with open(f, "r") as file:
            ref = json.load(open(f, "r"))
            if (
                ref.get(key, None) == search_value
                and ref.get("tokenized", None) == tokenized
            ):
                return ref
    return None


def get_source_ref(source_ref_path):
    with open(source_ref_path, "r") as file:
        return json.load(file)


def count_tokens(manifest_url, seqlen=2049):
    with (
        S3Path(manifest_url).open("r")
        if manifest_url.startswith("s3://")
        else open(manifest_url, "r")
    ) as f:
        manifest = [json.loads(line) for line in f]
    num_tokens = sum(int(line["num_sequences"]) for line in manifest) * seqlen
    return num_tokens


def get_s3_dir_size(dataset_path):
    bucket, prefix = dataset_path.replace("s3://", "").split("/", 1)
    total_size = 0
    for i, obj in enumerate(
        boto3.resource("s3").Bucket(bucket).objects.filter(Prefix=prefix)
    ):
        total_size += obj.size
    return total_size


def get_dir_size(dataset_path):
    total_size = 0
    for filename in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, filename)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    return total_size


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    dcnlp_commit_hash = repo.head.object.hexsha
    dcnlp_diff = repo.git.diff(repo.head.commit.tree)
    return dcnlp_commit_hash, dcnlp_diff


def generate_untokenized_dataset_json(
    args, source_refs, base_output_path, data_key=".json.zstd"
):
    sources = (
        [{"uuid": s["uuid"], "name": s["name"]} for s in source_refs]
        if source_refs
        else []
    )
    dcnlp_commit_hash, dcnlp_diff = get_git_info()

    dataset_json = {
        "uuid": str(uuid.uuid4().__str__()),
        "name": args.readable_name,
        "creation_date": datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        "dataset_url": os.path.join(base_output_path, "processed_data/"),
        "manifest_url": None,
        "sources": sources,
        "tokenized": False,
        "tokenizer": None,
        "num_tokens": None,
        "size": (
            get_s3_dir_size(args.output_dir)
            if args.output_dir.startswith("s3://")
            else get_dir_size(args.output_dir)
        ),
        "dcnlp_commit_hash": dcnlp_commit_hash,
        "dcnlp_diff": dcnlp_diff,
        "data_key": data_key,
    }

    return dataset_json


def generate_tokenized_dataset_json(args, source_refs, data_key="json.gz"):

    manifest_url = os.path.join(args.output.rstrip("/"), "manifest.jsonl")
    dcnlp_commit_hash, dcnlp_diff = get_git_info()
    sources = (
        [{"uuid": s["uuid"], "name": s["name"]} for s in source_refs]
        if source_refs
        else []
    )

    # TODO: Currently I just dump the entire yaml, is this the best thing to do?
    # Also, maybe would be nice to support automated generation of this yaml given input sources + weights
    sampling_yaml = None
    if args.do_sample:
        with open(args.default_dataset_yaml, "r") as file:
            sampling_yaml = yaml.safe_load(file)

    dataset_json = {
        "uuid": str(uuid.uuid4().__str__()),
        "name": args.readable_name,
        "creation_date": datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        "dataset_url": args.output,
        "manifest_url": manifest_url,
        "sources": sources,
        "tokenized": True,
        "tokenizer": args.tokenizer,
        "num_tokens": count_tokens(manifest_url, args.seqlen + 1),
        "size": (
            get_s3_dir_size(args.output_dir)
            if args.output_dir.startswith("s3://")
            else get_dir_size(args.output_dir)
        ),
        "dcnlp_commit_hash": dcnlp_commit_hash,
        "dcnlp_diff": dcnlp_diff,
        "data_key": data_key,
        "sampling_yaml": sampling_yaml,
    }

    return dataset_json
