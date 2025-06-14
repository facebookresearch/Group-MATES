# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
from transformers import AutoModel
from transformers.file_utils import ModelOutput
from huggingface_hub import PyTorchModelHubMixin

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class RelationalModel(nn.Module, PyTorchModelHubMixin):
    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        normlized: bool = True,
        sentence_pooling_method: str = "cls",
        temperature: float = 1.0,
        use_independent: bool = False,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        classifier_dropout = (
            self.model.config.classifier_dropout
            if self.model.config.classifier_dropout is not None
            else self.model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.mse = nn.MSELoss()

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temp = nn.Parameter(torch.tensor(temperature, requires_grad=True))
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.use_independent = use_independent
        self.config = self.model.config

    def encode(self, features):
        if features is None:
            return None
        psg_out = self.model(**features, return_dict=True)
        p_reps = psg_out.last_hidden_state[:, 0]
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        # p_reps = p_reps.reshape(-1, p_reps.size(1) * 4)
        p_reps = p_reps.reshape(-1, 4, p_reps.size(1)).mean(dim=1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(
        self,
        passage: Dict[str, Tensor] = None,
        label: Tensor = None,
    ):
        # (bs * num_rollouts, hidden_size)
        p_reps = self.encode(passage)

        pooled_output_p = self.dropout(p_reps)
        scores = self.classifier(pooled_output_p).squeeze()
        # (bs * num_rollouts, bs * num_rollouts)
        dependent_scores = self.compute_similarity(p_reps, p_reps) / self.temp

        N = scores.shape[0]
        mask = torch.tril(torch.ones(N, N, device=scores.device), diagonal=-1)
        penalties = (dependent_scores - 1) * mask

        penalty_sum = penalties.sum(dim=1)
        divisors = torch.arange(1, N, device=scores.device)
        scores[1:] = -scores[1:] * self.alpha * penalty_sum[1:] / divisors

        loss = self.compute_loss(scores, label)
        return EncoderOutput(
            loss=loss,
            scores=scores,
        )

    def compute_loss(self, scores, target):
        return self.mse(scores, target)
