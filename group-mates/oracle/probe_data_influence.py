# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from open_lm.distributed import world_info_from_env

from training.dataset_reference import DatasetReference
from training.file_utils import setup_logger
from training.hyperparameters import get_scale_config, Hyperparameters
from training.params import get_open_lm_args, parse_dcnlp_args

logger = setup_logger(__name__)


def process_dcnlp_args(args):
    """Helper script for setting up data reference, hparams, and name.

    Note: The reason this is a function is because it is used by other scripts (e.g. Sagemaker) to get the name from an
    args object.
    """
    data = None
    with open(args.data_config, "r") as f:
        data = DatasetReference(**json.load(f))

    # modify num tokens by multiplier
    hparams = None
    if args.re_evaluate:
        model_json = None
        with open(args.re_evaluate, "r") as f:
            model_json = json.load(f)
        hparams = Hyperparameters(**model_json["hyperparameters"])
        hparams.global_bs = 128
    else:
        hparams = get_scale_config(args.scale)

        # if argparse overrides scale config we should too
        # NOTE: this will be removed for public release but useful for grid search
        hparams.update_config(args)

    open_lm_args, name = get_open_lm_args(args, hparams, data)
    return open_lm_args, name, hparams, data


if __name__ == "__main__":
    args = parse_dcnlp_args()

    if args.clean_exp:
        assert (
            args.remote_sync is not None
        ), "must specify --remote-sync to use --clean-local-logs"

    open_lm_args, name, hparams, data = process_dcnlp_args(args)

    _, rank, world_size = world_info_from_env()
    if rank == 0:
        logger.info(f"Running training on scale: {args.scale}")
        logger.info(f"World size is {world_size}.")

    assert (
        hparams.global_bs % world_size == 0
    ), f"world size: {world_size} does not divide global batch size: {hparams.global_bs}"

    exp_data_models_path = Path(__file__).parent.parent / args.git_db
    if not exp_data_models_path.exists():
        os.makedirs(exp_data_models_path, exist_ok=True)

    model_path = exp_data_models_path / f"{name}.json"

    if not os.path.exists(os.path.join(args.logs, name)):
        # create this dir to prevent sync'ing errors
        os.makedirs(os.path.join(args.logs, name), exist_ok=True)

    if not args.skip_train:
        from .rollout_train import main

        print(f"Running with args:\n{open_lm_args}")

        main(open_lm_args)
