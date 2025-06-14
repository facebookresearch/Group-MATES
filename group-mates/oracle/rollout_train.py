# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import fsspec
import random
import functools
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from open_lm.params import parse_args
from open_lm.precision import get_autocast
from open_lm.model import create_model, Block
from open_lm.file_utils import pt_load, check_exists
from open_lm.losses import CrossEntropyLossWithZLoss
from datasets import Dataset, Features, Sequence, Value
from open_lm.distributed import init_distributed_device
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def load_model(args, model):
    checkpoint = pt_load(args.resume, map_location="cpu")
    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith("module"):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
        if "_orig_mod" in next(iter(sd.items()))[0]:
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    else:
        sd = checkpoint
    return sd


def load_optimizer(args, model, optimizer):
    potential_checkpoint = args.resume.replace("epoch_", "optimizer_")
    if check_exists(potential_checkpoint):
        checkpoint = pt_load(potential_checkpoint, map_location="cpu")
    else:
        checkpoint = pt_load(args.resume, map_location="cpu")
    osd = checkpoint["optimizer"]
    osd = FSDP.optim_state_dict_to_load(model, optimizer, osd)
    return osd


def train(model, optimizer, train_data, accumulation=False):
    optimizer.zero_grad()
    loss = CrossEntropyLossWithZLoss()
    autocast = get_autocast("amp_bfloat16")
    with autocast():
        inputs, targets = (
            train_data[:, :-1].contiguous().long().cuda(),
            train_data[:, 1:].contiguous().long().cuda(),
        )
        out, _, _ = model(inputs)
        total_loss = loss(out.reshape(-1, model.vocab_size), targets.reshape(-1))
    total_loss.backward()
    if isinstance(model, FSDP):
        model.clip_grad_norm_(1, norm_type=2.0)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2.0)
    if not accumulation:
        optimizer.step()


@torch.no_grad()
def evaluate(model, val_dataloader):
    model.eval()

    loss = torch.nn.CrossEntropyLoss(reduction="none")
    autocast = get_autocast("amp_bfloat16")
    with autocast():
        total_loss = 0.0
        cnt = 0
        for batch in val_dataloader:
            inputs, targets = batch["input_ids"][:, :-1], batch["labels"][:, 1:]
            out, _, _ = model(inputs)  # [bs, seq_len, vocab_size]
            targets = targets.reshape(-1)
            cur_loss = loss(out.reshape(-1, model.vocab_size), targets)
            total_loss += cur_loss[targets != -100].mean().item()
            cnt += 1

    model.train()
    return [total_loss / cnt]


def main(args):
    args = parse_args(args)
    args.resume = f"{os.environ.get('CKPT')}.pt"

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)
    random_seed(args.seed, 0)
    with torch.device(
        "meta" if args.experimental_meta_device and args.fsdp else args.device
    ):
        model = create_model(args)

    random_seed(args.seed, args.rank)
    if args.distributed:
        transformer_layer_cls = None

        transformer_layer_cls = {Block}
        transformer_auto_wrapper_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layer_cls,
        )
        # tries to follow gopher...
        mp_policy = None
        if args.fsdp_amp:
            print("=> using bfloat16 params as part of fsdp amp policy.")
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
            )
        elif args.fsdp_pure_bf16:
            print("=> using pure bfloat16 params as part of fsdp amp policy.")
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        if args.rank == 0:
            print(
                f"Before FSDP parameter num: {sum(p.numel() for p in model.parameters()):,}"
            )
            print(f"Before FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB")

        fsdp_kwargs = {}
        assert not (
            args.fsdp_hybrid and args.fsdp_hybrid_o2
        ), "Only --fsdp-hybrid or --fsdp-hybrid-o2 should be set."
        if args.fsdp_backward_prefetch:
            fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
        if args.fsdp_hybrid:
            fsdp_kwargs["sharding_strategy"] = ShardingStrategy.HYBRID_SHARD
        if args.fsdp_hybrid_o2:
            fsdp_kwargs["sharding_strategy"] = ShardingStrategy._HYBRID_SHARD_ZERO2
        print("=> FSDP kwargs: ", fsdp_kwargs)

        # Initialize FSDP. Use the same seed across workers to ensure reset_parameters is the same across workers.
        random_seed(args.seed, rank=0)
        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrapper_policy,
            device_id=device,
            mixed_precision=mp_policy,
            cpu_offload=CPUOffload(offload_params=args.fsdp_cpu_offload),
            use_orig_params=args.fsdp_use_orig_params,
            limit_all_gathers=args.fsdp_limit_all_gathers,
            **fsdp_kwargs,
        )

        print(
            f"After FSDP parameter num: {sum(p.numel() for p in model.parameters()):,} on rank {args.rank}"
        )
        print(
            f"After FSDP {torch.cuda.memory_allocated()/1024**3:.3} GB on rank {args.rank}"
        )

    sd = load_model(args, model)
    model.load_state_dict(sd)

    named_parameters = list(model.named_parameters())
    no_decay_params = []
    params = [p for n, p in named_parameters if p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": no_decay_params, "weight_decay": 0.0},
            {"params": params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    osd = load_optimizer(args, model, optimizer)

    probe_data = "group-mates/oracle/data/train.pt"
    of = fsspec.open(probe_data, "rb")
    with of as f:
        train_dataset = torch.load(f)
    dataset_len = len(train_dataset)
    print(dataset_len)

    def val_collate_fn(batch):
        input_ids = [torch.tensor(s["input_ids"], device="cuda") for s in batch]
        labels = [torch.tensor(s["labels"], device="cuda") for s in batch]

        x = pad_sequence(input_ids, batch_first=True, padding_value=0)
        y = pad_sequence(labels, batch_first=True, padding_value=-100)

        x = x[:, :2048]
        y = y[:, :2048]

        return {"input_ids": x, "labels": y}

    val_data = "group-mates/oracle/data/ref.pt"
    of = fsspec.open(val_data, "rb")
    with of as f:
        val_dataloader = DataLoader(
            torch.load(f)[:128],
            batch_size=64,
            collate_fn=val_collate_fn,
        )

    seed = int(os.environ.get("SEED"))
    print("SEED:", seed)
    np.random.seed(seed)
    left, right = seed * 5000, seed * 5000 + 5000
    try:
        bootstrap_indices = np.load(f"{os.environ.get('CKPT')}/bootstrap.npy")
    except:
        bootstrap_indices = None
    if bootstrap_indices is not None:
        ocache = f"{os.environ.get('CKPT')}/oracle-bootstrap/{seed}"
    else:
        ocache = f"{os.environ.get('CKPT')}/oracle-random/{seed}"
    oracle = []

    eval_base = evaluate(model, val_dataloader)[0]
    for i in tqdm(range(left, right)):
        model.load_state_dict(sd)
        optimizer.load_state_dict(osd)
        if bootstrap_indices is not None:
            probe_indices = bootstrap_indices[i]
        else:
            probe_indices = np.random.permutation(dataset_len)[:10]
        scores = []
        input_ids = []
        for probe_index in probe_indices:
            probe_data = train_dataset[probe_index]
            input_ids += probe_data[0][:-1].cpu().numpy().tolist()
            train(model, optimizer, probe_data)
            scores.append(evaluate(model, val_dataloader)[0] - eval_base)

        oracle.append(
            {
                "input_ids": input_ids,
                "scores": scores,
            }
        )
        if (i + 1) % 1000 == 0 or (i + 1) == right:
            if args.rank == 0:
                features = Features(
                    {
                        "input_ids": Sequence(Value("int32")),
                        "scores": Sequence(Value("float32")),
                    }
                )
                processed_ds = Dataset.from_list(oracle, features=features)
                processed_ds.save_to_disk(ocache)
