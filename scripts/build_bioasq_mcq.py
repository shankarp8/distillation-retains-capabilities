#!/usr/bin/env python
"""Convert BioASQ cloze questions into multiple choice (Section 4.1).

BioASQ ships expert-curated cloze-style questions. To score post-trained chat
models consistently with KUP, each question is turned into a 4-way multiple
choice item by generating three plausible distractors with GPT-5 and shuffling
them against the gold answer.

Usage
-----
    export OPENAI_API_KEY=...
    python scripts/build_bioasq_mcq.py --output data/bioasq_mcq.jsonl

The output is consumed by ``scripts/eval_domain.py --task bioasq``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from disc.data import write_jsonl  # noqa: E402

DISTRACTOR_PROMPT = """You are a careful biomedical distractor writer.
You will be given: (1) a question, (2) the correct answer, and (3) supporting context.
Return exactly THREE semantically plausible but wrong answer choices. The wrong
answer choices should be sufficiently close to the correct answer that a
knowledgeable test taker might confuse them.

Rules:
- Make them concise noun phrases (a few words).
- Same answer type/category as the correct answer.
- Must be clearly contradicted by or NOT supported by the context.
- Avoid negations like "not", "none of the above", or humorous/absurd options.
- Do not repeat the correct answer with trivial rephrasing.
- Keep them distinct from each other.

STRICT OUTPUT (JSON only):
{"distractors": ["<D1>", "<D2>", "<D3>"]}
"""

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*<context>", re.DOTALL | re.I)
CONTEXT_RE = re.compile(r"<context>\s*(.*)$", re.DOTALL | re.I)


def parse_answer_and_context(text: str) -> Tuple[str, str]:
    """Pull the gold answer and supporting context out of the packed text field."""
    ans = ANSWER_RE.search(text)
    ctx = CONTEXT_RE.search(text)
    answer = re.sub(r"\s+", " ", ans.group(1).strip()) if ans else ""
    context = re.sub(r"\s+", " ", ctx.group(1).strip()) if ctx else ""
    return answer, context


def extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0).strip() if match else text


def dedupe_distractors(distractors: List[str], correct: str) -> List[str]:
    """Drop distractors that collide with the gold answer or with each other."""
    seen = {correct.strip().lower()}
    unique = []
    for d in distractors:
        key = d.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(d.strip())
    return unique


def generate_distractors(client, model: str, question: str, answer: str,
                         context: str, retries: int = 3) -> Optional[List[str]]:
    payload = (
        f"Question: {question}\nCorrect answer: {answer}\n"
        f"Context: {context[:4000]}"
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DISTRACTOR_PROMPT},
                    {"role": "user", "content": payload},
                ],
            )
            raw = response.choices[0].message.content
            parsed = json.loads(extract_json_block(raw))
            distractors = dedupe_distractors(parsed.get("distractors", []), answer)
            if len(distractors) >= 3:
                return distractors[:3]
        except Exception as exc:  # noqa: BLE001 - transient API/parse failures
            print(f"  [retry {attempt + 1}/{retries}] {type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BioASQ multiple-choice questions")
    ap.add_argument("--output", default="data/bioasq_mcq.jsonl")
    ap.add_argument("--model", default="gpt-5", help="Distractor-generating model")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of questions")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    from datasets import load_dataset
    from openai import OpenAI

    client = OpenAI()
    rng = random.Random(args.seed)

    ds = load_dataset("kroshan/BioASQ")["test"]
    records: List[Dict] = []
    skipped = 0

    items = list(ds)[: args.limit] if args.limit else list(ds)
    for i, ex in enumerate(items):
        question = ex.get("question") or ""
        answer, context = parse_answer_and_context(ex["text"])
        if not (question and answer):
            skipped += 1
            continue

        distractors = generate_distractors(client, args.model, question, answer, context)
        if not distractors:
            skipped += 1
            continue

        options = distractors + [answer]
        rng.shuffle(options)

        records.append({
            "id": i,
            "question": question,
            "options": options,
            "answer": answer,
            "answer_index": options.index(answer) + 1,
            "context": context,
        })

        if (i + 1) % 25 == 0:
            print(f"  {len(records)} built / {i + 1} seen")
            write_jsonl(records, args.output)

    write_jsonl(records, args.output)
    print(f"[done] {len(records)} questions ({skipped} skipped) -> {args.output}")


if __name__ == "__main__":
    main()
