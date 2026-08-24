#!/usr/bin/env python
"""Generate the on-policy data that two baselines require.

Both are vLLM batch-generation jobs over the adaptation corpus that write a
.jsonl of ``{id, evidence, generations}``; they differ only in the prompt and
in how the output is consumed downstream.

  ``--mode rephrase``  Ask the initial model to paraphrase each document. Used
                       by ``+Rephrase`` (Yang et al., 2024b), which finetunes
                       on these on-policy rewrites instead of the raw text.

  ``--mode transfer``  Ask the initial model to continue each document. Used by
                       CD-base (Padmanabhan et al., 2023), whose KL is computed
                       over these generated continuations. DiSC needs no such
                       pass — it takes its suffixes from the document itself,
                       which is what makes it the cheaper algorithm.

Usage
-----
    python scripts/build_generations.py --mode rephrase --model qwen2.5-7b \
        --corpus kup --output data/rephrased_qwen7b_kup.jsonl

    python scripts/build_generations.py --mode transfer --model qwen3-8b \
        --corpus bioasq --output data/transfer_set_qwen8b_bioasq.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from disc.data import load_corpus  # noqa: E402
from disc.models import load_tokenizer, resolve_model  # noqa: E402

REPHRASE_SYSTEM = "You are a helpful writing assistant."
REPHRASE_USER = (
    "Below is an article. Rewrite it in your own words, preserving all factual "
    "content and named entities exactly.\n\n{document}\n\nRewrite:"
)

TRANSFER_SYSTEM = "You are a helpful writing assistant."
TRANSFER_USER = "Below is an article.\n\n{document}\n\nContinuation:"

PROMPTS = {
    "rephrase": (REPHRASE_SYSTEM, REPHRASE_USER),
    "transfer": (TRANSFER_SYSTEM, TRANSFER_USER),
}


def build_prompt(tokenizer, mode: str, document: str) -> str:
    system, user = PROMPTS[mode]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user.format(document=document)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build rephrase / transfer-set data")
    ap.add_argument("--mode", required=True, choices=["rephrase", "transfer"])
    ap.add_argument("--model", required=True, help="The initial post-trained model (on-policy)")
    ap.add_argument("--corpus", default="kup", choices=["kup", "bioasq"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n", type=int, default=1, help="Generations per document")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--start", type=int, default=0, help="Resume from this document index")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tokenizer = load_tokenizer(args.model)

    # Not shuffled: the transfer set is indexed by position in the corpus, and
    # training must see the same order.
    documents = load_corpus(args.corpus, shuffle=False)
    documents = documents[args.start:]
    print(f"[data] {len(documents)} documents from {args.corpus} (start={args.start})")

    llm = LLM(
        model=resolve_model(args.model),
        dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.start else "w"

    written = 0
    with open(args.output, mode, encoding="utf-8") as f_out:
        for start in range(0, len(documents), args.batch_size):
            chunk = documents[start:start + args.batch_size]
            prompts = [build_prompt(tokenizer, args.mode, d) for d in chunk]
            results = llm.generate(prompts, sampling)

            for offset, (document, res) in enumerate(zip(chunk, results)):
                record = {
                    "id": args.start + start + offset,
                    "evidence": document,
                    "generations": [o.text for o in res.outputs],
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            f_out.flush()
            print(f"  wrote {written}/{len(documents)}")

    print(f"[done] {written} records -> {args.output}")


if __name__ == "__main__":
    main()
