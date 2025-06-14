# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 mlfoundations
#
# This code is based on https://github.com/mlfoundations/dclm

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from time import gmtime, strftime

import requests
import yaml
from requests.structures import CaseInsensitiveDict


def submit_to_slack(filename):
    data = json.load(open(filename))
    score = data.get("aggregated_centered_results", -1)
    low_var_score = data.get("low_variance_datasets", -1)
    mmlu = data["eval_metrics"]["icl"].get("mmlu_fewshot", -1)
    name = data["name"]
    model = data["model"]
    uuid = data["uuid"]
    url = f"https://github.com/mlfoundations/dcnlp/tree/main/{filename}"

    message = f"New submission ({model}). Low Variance Score: {low_var_score:.4f}., Aggregated centered score: {score:.4f}. MMLU 5-shot Score: {mmlu: .4f}. Name: {name}. UUID: {uuid}. Full results at {url}"

    root = "hooks.slack.com"
    part1 = "T01AEJ66KHV"
    part2 = "B06HC24QGSG"
    part3 = "XOWNfYwTRmPzf57owBbFfw5t"
    url = f"https://{root}/services/{part1}/{part2}/{part3}"

    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    data = json.dumps({"text": message})
    resp = requests.put(url, headers=headers, data=data)

    return resp


if __name__ == "__main__":
    files = sys.argv[1].split()
    print(f"Starting submission for files: {files}")
    for file in files:
        if not os.path.exists(file):
            print(f"Skipping {file} because it does not exist")
            continue
        submit_to_slack(file)
