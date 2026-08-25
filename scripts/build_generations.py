#!/usr/bin/env python
"""Generate the on-policy data that two baselines require.

Both are vLLM batch-generation jobs over the (unshuffled) adaptation corpus that
write a .jsonl of ``{id, evidence, generations}``.

  ``--mode rephrase``  Ask the initial model to rewrite each document. Used by
                       ``+Rephrase`` (Yang et al., 2024b). Two prompt variants
                       exist in the original tree and both are kept verbatim:

                         --prompt rewrite   transfer_set.py       (rephrased_7b.jsonl, KUP)
                         --prompt rephrase  transfer_set_new2.py  (rephrased_3b_asq.jsonl, BioASQ)

  ``--mode transfer``  Ask the initial model to continue each document. Used by
                       CD-base (Padmanabhan et al., 2023). NOTE: the script that
                       produced the transfer_set_qwen8b_{asq,kup_new}.jsonl files
                       used for the paper is not in the original tree, so this
                       prompt could not be verified against it. Override it with
                       ``--user_prompt`` if the original is recovered.

Sampling (all originals): temperature 1.0, top_p 0.95, max_tokens 10000, n=1,
vLLM dtype "half", gpu_memory_utilization 0.9.

Usage
-----
    python scripts/build_generations.py --mode rephrase --prompt rewrite \
        --model qwen2.5-7b --corpus kup --output data/rephrased_qwen7b_kup.jsonl

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

SYSTEM_PROMPT = "You are a helpful writing assistant."

# Verbatim, including the double space in the "rewrite" variant and the
# "Continuation:" cue that both rephrase prompts end with.
USER_PROMPTS = {
    "rewrite": "Below is a article. Rewrite it in your own words  \n\n{document}\n\nContinuation:",
    "rephrase": "Below is a article. Rephrase it in your own words.\n\n{document}\n\nContinuation:",
    # Unverified (see module docstring).
    "transfer": "Below is a article.\n\n{document}\n\nContinuation:",
}

DEFAULT_PROMPT_FOR_MODE = {"rephrase": "rewrite", "transfer": "transfer"}


def build_prompt(tokenizer, user_template: str, document: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_template.format(document=document)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build rephrase / transfer-set data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", required=True, choices=["rephrase", "transfer"])
    ap.add_argument("--model", required=True, help="The initial post-trained model (on-policy)")
    ap.add_argument("--corpus", default="kup", choices=["kup", "bioasq"])
    ap.add_argument("--output", required=True)
    ap.add_argument("--prompt", default=None, choices=sorted(USER_PROMPTS),
                    help="Which original prompt to use (default: rewrite for --mode rephrase, "
                         "transfer for --mode transfer)")
    ap.add_argument("--user_prompt", default=None,
                    help="Custom user message template containing {document}; overrides --prompt")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--n", type=int, default=1, help="Generations per document")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=10000)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--start", type=int, default=0, help="Resume from this document index")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    prompt_name = args.prompt or DEFAULT_PROMPT_FOR_MODE[args.mode]
    user_template = args.user_prompt or USER_PROMPTS[prompt_name]
    print(f"[prompt] {prompt_name if not args.user_prompt else 'custom'}: {user_template!r}")

    tokenizer = load_tokenizer(args.model)

    # Not shuffled: the transfer set is indexed by position in the corpus, and
    # training must see the same order.
    documents = load_corpus(args.corpus, shuffle=False)
    documents = documents[args.start:]
    print(f"[data] {len(documents)} documents from {args.corpus} (start={args.start})")

    llm = LLM(
        model=resolve_model(args.model),
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n,
        stop=None,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.start else "w"

    written = 0
    with open(args.output, mode, encoding="utf-8") as f_out:
        for start in range(0, len(documents), args.batch_size):
            chunk = documents[start:start + args.batch_size]
            prompts = [build_prompt(tokenizer, user_template, d) for d in chunk]
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
