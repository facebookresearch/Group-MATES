# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import numpy as np
import datasets
import torch
import sys
import os

from modeling_data_influence_model import RelationalModel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    def __call__(self, features):
        passage = [f["passage"] for f in features]
        score = torch.tensor([f["score"] for f in features], dtype=torch.float32)
        bs, num_rollouts = score.shape

        if isinstance(passage[0], list):
            passage = sum(passage, [])

        d_collated = self.tokenizer.batch_encode_plus(
            passage,
            max_length=2048,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        for key, value in d_collated.items():
            d_collated[key] = value.reshape(bs * num_rollouts * 4, -1)

        return {"passage": d_collated, "label": score.reshape(-1)}


if __name__ == "__main__":
    stage = sys.argv[1]
    model_name = "BAAI/bge-base-en-v1.5"
    model = RelationalModel(model_name=model_name)

    args = TrainingArguments(
        f"{os.environ.get('CKPT')}/dim-{stage}",
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        num_train_epochs=5,
        warmup_steps=50,
        logging_steps=5,
        eval_steps=50,
        save_steps=50,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        bf16=True,
        report_to="wandb",
        run_name=f"relational-data-influence-model",
        remove_unused_columns=False,
    )

    pythia_tokenizer = AutoTokenizer.from_pretrained(
        "togethercomputer/RedPajama-INCITE-Base-7B-v0.1"
    )
    if stage == "init":
        dataset = datasets.concatenate_datasets(
            [
                datasets.load_from_disk(f"{os.environ.get('CKPT')}/oracle-random/{i}")
                for i in range(4)
            ]
        )
    elif stage == "final":
        dataset = datasets.concatenate_datasets(
            [
                datasets.load_from_disk(f"{os.environ.get('CKPT')}/oracle-random/{i}")
                for i in range(4)
            ]
            + [
                datasets.load_from_disk(f"{os.environ.get('CKPT')}/oracle-bootstrap/{i}")
                for i in range(4)
            ]
        )
    else:
        raise ValueError("Invalid stage. Use 'init' or 'final'.")

    # Transform scores by subtracting the previous one, i.e., scores[i] = scores[i] - scores[i-1], scores[0] remains scores[0]
    def transform_scores_batched(examples):
        new_scores_batch = []
        for scores_list in examples["scores"]:
            scores = scores_list.copy()  # Make a copy
            for i in range(1, len(scores)):
                scores[i] = scores[i] - scores[i - 1]
            new_scores_batch.append(scores)
        return {"scores": new_scores_batch}

    dataset = dataset.map(transform_scores_batched, batched=True, num_proc=8)
    mean_value = np.mean(np.array(dataset["scores"]))
    std_value = np.std(np.array(dataset["scores"]))
    num_rollouts = 10
    print(mean_value, std_value)

    def preprocess_data(examples):
        passages = [
            [input_ids[i : i + 2048] for i in range(0, num_rollouts * 2048, 2048)]
            for input_ids in examples["input_ids"]
        ]
        passages = [
            pythia_tokenizer.batch_decode(
                passage_list,
                skip_special_tokens=True,
            )
            for passage_list in passages
        ]
        scores = [
            [(s[i] - mean_value) / std_value for i in range(num_rollouts)]
            for s in examples["scores"]
        ]
        return {"passage": passages, "score": scores}

    dataset = dataset.map(
        preprocess_data,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.train_test_split(test_size=1000, seed=1234, shuffle=True)
    train_dataset = dataset["train"]
    print("Training data size:", len(train_dataset))
    eval_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = EmbedCollator(tokenizer)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        pearson_corr = pearsonr(predictions, labels)[0]
        spearman_corr = spearmanr(predictions, labels)[0]
        return {
            "mse": mean_squared_error(labels, predictions),
            "mae": mean_absolute_error(labels, predictions),
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()
    trainer.save_model()

    # Evaluate the best model
    eval_results = trainer.evaluate()

    # Print the evaluation results
    print("Best evaluation results:", eval_results)
