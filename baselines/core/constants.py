# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 mlfoundations
#
# This code is based on https://github.com/mlfoundations/dclm

# JSONL keys
CONTENT = "text"
URL = "url"
CHUNK = "chunk"

# Stats file keys
PROCESS_SETUP_KEY_NAME = 'process_setup'
PROCESS_END_KEY_NAME = 'process_finished'
COMMIT_KEY_NAME = 'commit'

GLOBAL_FUNCTIONS = {
	'exact_dedup': None,
} 