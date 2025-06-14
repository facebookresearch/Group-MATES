# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2024 mlfoundations
#
# This code is based on https://github.com/mlfoundations/dclm

"""
This script assigns a given text a dim score.
"""

import os
from typing import Callable, Dict, List

from core.constants import CONTENT
from core.factory_utils import factory_function

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
import numpy as np
import torch
from transformers import AutoTokenizer

from .modeling_data_influence_model import BertForSequenceClassification

rng = np.random.default_rng()


def load_dim(model_filename):
    tokenizer = AutoTokenizer.from_pretrained(
        model_filename,
        max_length=2048,
        padding="max_length",
    )

    model = BertForSequenceClassification.from_pretrained(
        model_filename,
        torch_dtype=torch.bfloat16,
        problem_type="regression",
        num_labels=1,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model.to(device)

    return model, tokenizer


@torch.no_grad()
def assign_dim_score(
    model: BertForSequenceClassification,
    tokenizer: AutoTokenizer,
    content: str,
) -> dict:
    """
    This function assigns a given text a dim score.

    Parameters:
    model (BertForSequenceClassification): The data influence model to use for the classification.
    content (str): The text to assign.

    Returns:
    dim score.
    """
    # Clean the input text by joining all lines into a single string
    text = " ".join(content.strip().splitlines())

    # Make the prediction
    example = tokenizer.batch_encode_plus(
        [text],
        max_length=2048,
        padding="max_length",
        truncation=True,
    )
    outputs = model(
        torch.tensor(example["input_ids"], device="cuda"),
        attention_mask=torch.tensor(example["attention_mask"], device="cuda"),
        token_type_ids=torch.tensor(example["token_type_ids"], device="cuda"),
    )

    # Return the output (a score)
    return outputs.logits.detach().float().cpu().numpy()[0][0]


@factory_function
def assign_dim_score_enricher(
    model_filename="",
    key: str = "dim_score",
    overwrite: bool = False,
) -> Callable[[Dict], List[Dict]]:
    """
    Assigns the given page with the dim score.
 
    Parameters:
        page (dict): The page to enrich.
        model_filename (str): The name of the fasttext model file. Assumes it is placed in MODEL_SUBDIRECTORY.
        key (str): The key to store the text type under.
        overwrite (bool): Whether to overwrite the existing value of the key.

    Returns:
        A function that assigns the given page with the dim score.
    """
    model, tokenizer = load_dim(model_filename)

    def enrich(page: Dict) -> List[Dict]:
        assert overwrite or key not in page, f"cannot overwrite an existing key {key}"
        page[key] = assign_dim_score(model, tokenizer, page[CONTENT])
        return [page]

    return enrich
