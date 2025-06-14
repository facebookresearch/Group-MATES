# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 mlfoundations
#
# This code is based on https://github.com/mlfoundations/dclm

from .dedup_jsonl import dedup_jsonl
from baselines.core.constants import GLOBAL_FUNCTIONS

GLOBAL_FUNCTIONS["exact_dedup"] = dedup_jsonl
