#!/usr/bin/env python
"""Knowledge-adaptation evaluation (T_D): KUP and BioASQ multiple choice.

Both datasets are scored the same way: sample ``--n`` answers per question at
temperature 1.0 and take the majority vote, which cuts variance on these small
evaluation sets. Generation runs through vLLM.

Usage
-----
    python scripts/eval_domain.py --task kup \
        --model_path runs/qwen7b_kup_disc/step_5000 \
        --data data/wikimcq_df.pickle

    python scripts/eval_domain.py --task bioasq \
        --model_path runs/qwen7b_bioasq_disc/step_3264 \
        --data data/bioasq_mcq.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from disc.data import load_jsonl  # noqa: E402

# --------------------------------------------------------------------------
# Prompts (Appendix D)
# --------------------------------------------------------------------------

KUP_TEMPLATE = """You will answer a multiple-choice question about {entity}.
- Put your reasoning in: <think> ... </think>
- Put ONLY the final answer letter (A/B/C/D) in: <answer> ... </answer>
- Do not include any other text outside these tags.

{question}
"""

BIOASQ_TEMPLATE = """Answer the multiple-choice question by reasoning briefly in <think>...</think> and then giving ONLY the numeral of the correct option inside <answer>...</answer>.

Question:
{question}

Options:
{options}

Rules:
 - Put your final choice as a numeral only (e.g., 2) inside <answer> tags.
 - Do NOT repeat the option text inside <answer>.
 - Choose exactly one option.
"""

# KUP entities with malformed questions, excluded from scoring.
KUP_EXCLUDED_ENTITY_IDS = [4641, 7759, 1868, 234]

# --------------------------------------------------------------------------
# Answer parsing
# --------------------------------------------------------------------------

LETTER_TAG_RE = re.compile(r"<answer>\s*([ABCD])\s*</answer>", re.I | re.S)
NUMERAL_TAG_RE = re.compile(r"<answer>\s*([1-9][0-9]*)\s*</answer>", re.I | re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.I | re.S)


def strip_think(text: str) -> str:
    return THINK_RE.sub("", text) if isinstance(text, str) else ""


def parse_letter(response: str) -> Tuple[Optional[str], str]:
    """Extract an A-D answer, falling back through progressively looser parsers.

    Returns ``(answer, parser_name)`` so the caller can report how often the
    model actually honoured the tag format.
    """
    if not isinstance(response, str) or not response.strip():
        return None, "none"

    matches = LETTER_TAG_RE.findall(response)
    if matches:
        return matches[-1].upper(), "tags"

    cleaned = strip_think(response)

    for line in cleaned.splitlines():
        if "answer" in line.lower():
            m = re.search(r"answer\s*[:\-]?\s*\(?([abcd])\b", line.lower())
            if m:
                return m.group(1).upper(), "literal"

    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped and stripped[0].upper() in "ABCD":
            if len(stripped) == 1 or stripped[1] in {":", ".", ")", " ", "-"}:
                return stripped[0].upper(), "first_line"

    return None, "none"


def parse_numeral(response: str, k: int) -> Tuple[Optional[int], str]:
    """Extract a 1..k option index (BioASQ scoring)."""
    if not isinstance(response, str) or not response.strip():
        return None, "none"

    matches = NUMERAL_TAG_RE.findall(response)
    if matches:
        n = int(matches[-1])
        if 1 <= n <= k:
            return n, "tags"

    cleaned = strip_think(response)
    m = re.search(r"\b([1-9][0-9]*)\b", cleaned)
    if m:
        n = int(m.group(1))
        if 1 <= n <= k:
            return n, "fallback_numeral"

    m = re.search(r"\b([A-Z])\b", cleaned)
    if m:
        n = ord(m.group(1).upper()) - ord("A") + 1
        if 1 <= n <= k:
            return n, "fallback_letter"

    return None, "none"


def majority_vote(values: List) -> Optional[object]:
    series = pd.Series([v for v in values if v is not None])
    if series.empty:
        return None
    mode = series.mode()
    return mode[0] if not mode.empty else None


# --------------------------------------------------------------------------
# Task loaders
# --------------------------------------------------------------------------


def build_kup(data_path: str):
    df = pd.read_pickle(data_path).dropna().reset_index(drop=True)
    df = df[~df["entity_id"].isin(KUP_EXCLUDED_ENTITY_IDS)].reset_index(drop=True)
    prompts = [
        KUP_TEMPLATE.format(entity=row["entity"], question=row["question"])
        for _, row in df.iterrows()
    ]
    golds = list(df["answer"])
    ids = list(df["entity_id"])
    return prompts, golds, ids, None


def build_bioasq(data_path: str):
    records = load_jsonl(data_path)
    prompts, golds, ids, n_options = [], [], [], []
    for i, ex in enumerate(records):
        options = list(ex["options"])
        options_str = "\n".join(f"{j + 1}. {opt}" for j, opt in enumerate(options))
        prompts.append(BIOASQ_TEMPLATE.format(question=ex["question"], options=options_str))
        # Gold is stored either as the answer text or as a 1-indexed position.
        if "answer_index" in ex:
            golds.append(int(ex["answer_index"]))
        else:
            golds.append(options.index(ex["answer"]) + 1)
        ids.append(ex.get("id", i))
        n_options.append(len(options))
    return prompts, golds, ids, n_options


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Domain adaptation evaluation")
    ap.add_argument("--task", required=True, choices=["kup", "bioasq"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True, help="wikimcq_df.pickle (KUP) or bioasq_mcq.jsonl")
    ap.add_argument("--output_dir", default=None, help="Default: {model_path}/{task}_eval")
    ap.add_argument("--n", type=int, default=5, help="Samples per question for majority vote")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    builder = {"kup": build_kup, "bioasq": build_bioasq}[args.task]
    prompts, golds, ids, n_options = builder(args.data)
    print(f"[data] {len(prompts)} questions for {args.task}")

    output_dir = args.output_dir or os.path.join(args.model_path, f"{args.task}_eval")
    os.makedirs(output_dir, exist_ok=True)

    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, n=args.n)

    outputs = llm.generate(prompts, sampling)

    rows = []
    for prompt_idx, out in enumerate(outputs):
        for sample_idx, o in enumerate(out.outputs):
            if args.task == "kup":
                answer, method = parse_letter(o.text)
            else:
                answer, method = parse_numeral(o.text, n_options[prompt_idx])
            rows.append({
                "question_id": ids[prompt_idx],
                "sample_idx": sample_idx,
                "prompt": prompts[prompt_idx],
                "response": o.text,
                "model_answer": answer,
                "parse_method": method,
                "gold_answer": golds[prompt_idx],
            })

    raw = pd.DataFrame(rows)
    raw["correct"] = raw["model_answer"] == raw["gold_answer"]

    summary = (
        raw.groupby("question_id")
        .agg(
            gold_answer=("gold_answer", "first"),
            voted_answer=("model_answer", lambda x: majority_vote(list(x))),
            confidence=("correct", "mean"),
        )
        .reset_index()
    )
    summary["correct"] = summary["voted_answer"] == summary["gold_answer"]

    raw.to_pickle(os.path.join(output_dir, "raw_responses.pickle"))
    summary.to_pickle(os.path.join(output_dir, "summary_results.pickle"))

    accuracy = float(summary["correct"].mean())
    unparsed = int(summary["voted_answer"].isna().sum())
    metrics = {
        "task": args.task,
        "model_path": args.model_path,
        "accuracy": accuracy,
        "n_questions": len(summary),
        "unparsed": unparsed,
        "n_samples": args.n,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{args.task} accuracy: {accuracy * 100:.2f}  ({unparsed} unparsed)")
    print("parse methods:\n" + raw["parse_method"].value_counts().to_string())
    print(f"saved -> {output_dir}")


if __name__ == "__main__":
    main()
