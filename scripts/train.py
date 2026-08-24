#!/usr/bin/env python
"""Continual knowledge adaptation — all methods, one entry point.

Examples
--------
Standard finetuning on KUP:
    python scripts/train.py --method ft --model qwen2.5-7b --corpus kup \
        --lr 1e-5 --save_dir runs/qwen7b_kup_ft

FT + KL regularization (beta = 0.1):
    python scripts/train.py --method kl --model qwen2.5-7b --corpus kup \
        --lr 1e-5 --kl_beta 0.1 --save_dir runs/qwen7b_kup_kl

FT + LoRA (r=16):
    python scripts/train.py --method lora --model qwen3-8b --corpus bioasq \
        --lr 5e-5 --save_dir runs/qwen3_8b_bioasq_lora

FT + Rephrase (on-policy self-distillation):
    python scripts/train.py --method rephrase --model llama3.1-8b --corpus bioasq \
        --rephrase_path data/rephrased_llama8b_bioasq.jsonl \
        --lr 1e-6 --save_dir runs/llama8b_bioasq_rephrase

FT + TALR:
    python scripts/train.py --method talr --model qwen2.5-7b --corpus kup \
        --lr 1e-5 --save_dir runs/qwen7b_kup_talr

CD-base (transfer-set context distillation):
    python scripts/train.py --method cd_base --model qwen3-8b --corpus bioasq \
        --transfer_path data/transfer_set_qwen8b_bioasq.jsonl \
        --lr 4e-6 --save_dir runs/qwen3_8b_bioasq_cdbase

DiSC:
    python scripts/train.py --method disc --model qwen2.5-7b --corpus kup \
        --lr 3e-6 --softmax_temp 2.0 --num_splits 5 \
        --save_dir runs/qwen7b_kup_disc

DiSC split-strategy ablation:
    python scripts/train.py --method disc --split_strategy token_random \
        --model qwen2.5-3b --corpus kup --lr 3e-6 --save_dir runs/ablation_token_random
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from disc.data import load_corpus, load_replay, load_transfer_map  # noqa: E402
from disc.models import apply_lora, load_frozen_teacher, load_model, load_tokenizer  # noqa: E402
from disc.splits import STRATEGIES  # noqa: E402
from disc.trainer import ALL_METHODS, TrainConfig, Trainer, set_seed  # noqa: E402

# Methods that need a frozen copy of the initial post-trained model.
NEEDS_TEACHER = {"kl", "disc", "cd_base"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continual knowledge adaptation of post-trained LMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    core = p.add_argument_group("core")
    core.add_argument("--method", required=True, choices=ALL_METHODS)
    core.add_argument("--model", dest="model_name", default="qwen2.5-7b",
                      help="Alias (see disc.models.MODEL_ALIASES), HF id, or local path")
    core.add_argument("--corpus", default="kup", choices=["kup", "bioasq"])
    core.add_argument("--save_dir", required=True)
    core.add_argument("--lr", type=float, required=True)
    core.add_argument("--seed", type=int, default=1234)
    core.add_argument("--save_every", type=int, default=0,
                      help="Save an intermediate checkpoint every N steps (0 = final only)")
    core.add_argument("--max_steps", type=int, default=0, help="0 = one full epoch")
    core.add_argument("--max_length", type=int, default=None)
    core.add_argument("--weight_decay", type=float, default=0.01)
    core.add_argument("--device", default="cuda:0")
    core.add_argument("--amp_dtype", default="bfloat16", choices=["bfloat16", "float16"])
    core.add_argument("--no_shuffle", action="store_true")

    kl = p.add_argument_group("+KL")
    kl.add_argument("--kl_beta", type=float, default=0.1)
    kl.add_argument("--kl_temperature", type=float, default=1.0)

    talr = p.add_argument_group("+TALR")
    talr.add_argument("--talr_tau", type=float, default=None,
                      help="Fixed tau; omit to recompute the dynamic tau each step")
    talr.add_argument("--talr_normalize", default="mean", choices=["mean", "sum"])

    lora = p.add_argument_group("+LoRA")
    lora.add_argument("--lora_rank", type=int, default=16)
    lora.add_argument("--lora_alpha", type=int, default=32)
    lora.add_argument("--lora_dropout", type=float, default=0.1)

    cd = p.add_argument_group("context distillation")
    cd.add_argument("--softmax_temp", type=float, default=2.0,
                    help="Distillation temperature (2.0 for DiSC, 1.0 for CD-base)")
    cd.add_argument("--num_splits", type=int, default=5, help="|I|, split points per document")
    cd.add_argument("--split_strategy", default="sentence_boundary", choices=sorted(STRATEGIES))
    cd.add_argument("--suffix_tokens", type=int, default=0,
                    help="Fixed suffix length for token_random_variable_suffix")
    cd.add_argument("--transfer_path", default=None, help="Transfer set .jsonl for --method cd_base")
    cd.add_argument("--rephrase_path", default=None, help="Rephrases .jsonl for --method rephrase")

    replay = p.add_argument_group("replay")
    replay.add_argument("--replay_source", default=None,
                        help="gsm8k | math | alpaca | path to .jsonl")
    replay.add_argument("--replay_n", type=int, default=0)
    replay.add_argument("--replay_kl_weight", type=float, default=0.0,
                        help="If > 0, apply KL to the initial policy on replay examples only")

    return p.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.method == "rephrase" and not args.rephrase_path:
        raise SystemExit(
            "--method rephrase requires --rephrase_path\n"
            "  build it with: scripts/build_generations.py --mode rephrase"
        )
    if args.method == "cd_base" and not args.transfer_path:
        raise SystemExit(
            "--method cd_base requires --transfer_path\n"
            "  build it with: scripts/build_generations.py --mode transfer"
        )
    if args.replay_kl_weight > 0 and not args.replay_source:
        raise SystemExit("--replay_kl_weight requires --replay_source")


def main() -> None:
    args = parse_args()
    validate(args)
    set_seed(args.seed)

    cfg = TrainConfig(
        method=args.method,
        model_name=args.model_name,
        corpus=args.corpus,
        save_dir=args.save_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        save_every=args.save_every,
        max_steps=args.max_steps,
        max_length=args.max_length,
        kl_beta=args.kl_beta,
        kl_temperature=args.kl_temperature,
        talr_tau=args.talr_tau,
        talr_normalize=args.talr_normalize,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        rephrase_path=args.rephrase_path,
        softmax_temp=args.softmax_temp,
        num_splits=args.num_splits,
        split_strategy=args.split_strategy,
        suffix_tokens=args.suffix_tokens,
        transfer_path=args.transfer_path,
        replay_source=args.replay_source,
        replay_n=args.replay_n,
        replay_kl_weight=args.replay_kl_weight,
        device=args.device,
        amp_dtype=args.amp_dtype,
    )

    print(f"[config] method={cfg.method} model={cfg.model_name} corpus={cfg.corpus} lr={cfg.lr}")

    tokenizer = load_tokenizer(cfg.model_name)

    # CD-base indexes the transfer set by position in the *unshuffled* corpus,
    # so the document order must match the order used at generation time.
    shuffle = not args.no_shuffle and cfg.method != "cd_base"
    documents = load_corpus(
        cfg.corpus,
        rephrase_path=cfg.rephrase_path,
        shuffle=shuffle,
        seed=cfg.seed,
    )
    print(f"[data] {len(documents)} documents")

    replay_flags = None
    if cfg.replay_source and cfg.replay_n > 0:
        replay_docs = load_replay(cfg.replay_source, cfg.replay_n, seed=cfg.seed)
        replay_flags = [False] * len(documents) + [True] * len(replay_docs)
        documents = documents + replay_docs
        order = list(range(len(documents)))
        import random as _random
        _random.Random(cfg.seed).shuffle(order)
        documents = [documents[i] for i in order]
        replay_flags = [replay_flags[i] for i in order]
        print(f"[data] mixed in {len(replay_docs)} replay demonstrations from {cfg.replay_source}")

    model = load_model(cfg.model_name, device=cfg.device)
    if cfg.method == "lora":
        model = apply_lora(model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout)
        model.to(cfg.device)
    model.train()

    teacher = None
    if cfg.method in NEEDS_TEACHER or cfg.replay_kl_weight > 0:
        teacher = load_frozen_teacher(cfg.model_name, device=cfg.device, dtype=torch.bfloat16)
        print("[model] loaded frozen teacher")

    transfer_map = load_transfer_map(cfg.transfer_path) if cfg.method == "cd_base" else None

    trainer = Trainer(
        cfg, model, tokenizer, documents,
        teacher=teacher, replay_flags=replay_flags, transfer_map=transfer_map,
    )
    final = trainer.train()
    print(f"[done] final checkpoint: {final}")


if __name__ == "__main__":
    main()
