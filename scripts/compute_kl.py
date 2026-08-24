#!/usr/bin/env python
"""Average per-token KL between the initial and adapted models (Section 6.2).

Shenfeld et al. (2025) report that per-token KL on the training data is the
best predictor of downstream forgetting. This script measures that quantity so
it can be correlated against measured forgetting on IFEval and MATH.

Usage
-----
    python scripts/compute_kl.py --base_model qwen2.5-7b --corpus kup \
        --checkpoints runs/qwen7b_kup_ft/step_5000 runs/qwen7b_kup_kl/step_5000 \
        --output results/kl_kup_qwen7b.json

Results are appended incrementally, so the script can be re-run to add
checkpoints without recomputing finished ones.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from disc.data import EvidenceDataset, load_corpus  # noqa: E402
from disc.models import load_tokenizer, resolve_model  # noqa: E402


@torch.no_grad()
def average_per_token_kl(model_p, model_q, dataloader, device_p, device_q, max_batches=None) -> float:
    """KL(p || q) per token, where p is the initial model and q the adapted one."""
    model_p.eval()
    model_q.eval()

    total_kl, total_tokens = 0.0, 0
    skipped = 0

    for step, batch in enumerate(tqdm(dataloader, desc="KL")):
        if max_batches is not None and step >= max_batches:
            break

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        logits_p = model_p(
            input_ids=input_ids.to(device_p), attention_mask=attention_mask.to(device_p)
        ).logits
        logits_q = model_q(
            input_ids=input_ids.to(device_q), attention_mask=attention_mask.to(device_q)
        ).logits.to(device_p)

        kl_token = F.kl_div(
            F.log_softmax(logits_q, dim=-1),
            F.log_softmax(logits_p, dim=-1),
            reduction="none",
            log_target=True,
        ).sum(dim=-1) * attention_mask.to(device_p)

        if torch.isnan(kl_token).any() or torch.isinf(kl_token).any():
            skipped += 1
            continue

        total_kl += kl_token.sum().item()
        total_tokens += attention_mask.sum().item()

    if skipped:
        print(f"[warn] skipped {skipped} batches with nan/inf")
    return total_kl / total_tokens if total_tokens else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-token KL vs. the initial policy")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--corpus", default="kup", choices=["kup", "bioasq"])
    ap.add_argument("--output", default="results/kl_results.json")
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--device_base", default="cuda:0")
    ap.add_argument("--device_ft", default="cuda:1")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.base_model)
    documents = load_corpus(args.corpus, shuffle=True, seed=args.seed)
    loader = DataLoader(
        EvidenceDataset(documents, tokenizer), batch_size=1, pin_memory=True, num_workers=2
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
    done = {r["model_path"] for r in results}

    base = AutoModelForCausalLM.from_pretrained(resolve_model(args.base_model)).to(args.device_base)

    for ckpt in args.checkpoints:
        if ckpt in done:
            print(f"[skip] {ckpt} (already computed)")
            continue

        ft = AutoModelForCausalLM.from_pretrained(ckpt).to(args.device_ft)
        kl_nats = average_per_token_kl(
            base, ft, loader,
            torch.device(args.device_base), torch.device(args.device_ft),
            max_batches=args.max_batches,
        )

        results.append({
            "model_path": ckpt,
            "base_model_path": args.base_model,
            "corpus": args.corpus,
            "kl_nats_per_token": round(kl_nats, 6),
            "kl_bits_per_token": round(kl_nats / math.log(2), 6),
        })
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

        print(f"{ckpt}: {kl_nats:.4f} nats/token ({kl_nats / math.log(2):.4f} bits)")

        del ft
        torch.cuda.empty_cache()

    print(f"\n[done] {args.output}")


if __name__ == "__main__":
    main()
