# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 mlfoundations
#
# This code is based on https://github.com/mlfoundations/dclm

#!/bin/bash

# Clone the repository
git clone https://github.com/mosaicml/llm-foundry.git

# Check if the clone was successful
if [ $? -eq 0 ]; then
    echo "Repository cloned successfully."
else
    echo "Failed to clone the repository."
    exit 1
fi

# Copy the directory to the current directory
cp -r llm-foundry/scripts/eval/local_data .

# Check if the copy was successful
if [ $? -eq 0 ]; then
    echo "Directory copied successfully."
else
    echo "Failed to copy the directory."
    exit 1
fi

# Remove the cloned repository
rm -rf llm-foundry

# Check if the delete was successful
if [ $? -eq 0 ]; then
    echo "Repository deleted successfully."
else
    echo "Failed to delete the repository."
    exit 1
fi