#!/usr/bin/env python
"""Convert BioASQ cloze questions into multiple choice (Section 4.1, Appendix D.2).

Port of ``asq_questions.py``, which built ``bioasq_mcq_harder_full.jsonl`` —
the file ``asq_eval.py`` / ``eval_vllm.py`` read. Everything that affects the
resulting item set is kept as it was:

  * questions come from the ``train`` split (the same documents the models
    are adapted on), iterated in reverse order, deduplicated by question text;
  * the distractor prompt is the one printed in Appendix D.2;
  * options are shuffled with ``random.seed(2025)`` on the global RNG;
  * records use the ``correct_index`` (0-based) schema with a ``meta`` block.

Usage
-----
    export OPENAI_API_KEY=...
    python scripts/build_bioasq_mcq.py --output data/bioasq_mcq_harder_full.jsonl

The output is consumed by ``scripts/eval_domain.py --task bioasq``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from typing import Any, Dict, List, Tuple

# -------------------- Helpers --------------------

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*<context>", flags=re.DOTALL | re.IGNORECASE)
CONTEXT_RE = re.compile(r"<context>\s*(.*)$", flags=re.DOTALL | re.IGNORECASE)


def parse_answer_and_context(text: str) -> Tuple[str, str]:
    """
    Extracts the gold answer (between <answer> ... <context>) and the context (after <context>).
    Returns (answer, context). Strips extra whitespace and collapses spaces.
    """
    ans_match = ANSWER_RE.search(text)
    ctx_match = CONTEXT_RE.search(text)

    gold_answer = ans_match.group(1).strip() if ans_match else ""
    context = ctx_match.group(1).strip() if ctx_match else ""

    gold_answer = re.sub(r"\s+", " ", gold_answer)
    context = re.sub(r"\s+", " ", context)
    return gold_answer, context


def _extract_json_block(text: str) -> str:
    """
    Try to extract the first valid-looking JSON object from the text.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    return text


def _ensure_unique_distractors(distractors: List[str], correct: str) -> List[str]:
    """
    Deduplicate, strip, remove empties, and drop anything equal (case-insensitive) to the correct answer.
    If less than 3 after filtering, pad with simple type-mismatched fillers.
    """
    norm_correct = correct.strip().lower()
    cleaned = []
    seen = set()
    for d in distractors:
        d = str(d).strip()
        if not d:
            continue
        if d.lower() == norm_correct:
            continue
        if d.lower() in seen:
            continue
        seen.add(d.lower())
        cleaned.append(d)
        if len(cleaned) == 3:
            break

    fillers = ["Not stated", "Unrelated", "Insufficient data"]
    i = 0
    while len(cleaned) < 3:
        fill = fillers[i % len(fillers)]
        if fill.lower() != norm_correct and fill.lower() not in seen:
            cleaned.append(fill)
            seen.add(fill.lower())
        i += 1
    return cleaned[:3]


# -------------------- GPT-5 Call --------------------

DISTRACTOR_SYSTEM_PROMPT = """You are a careful biomedical distractor writer.

You will be given: (1) a question, (2) the correct answer, and (3) supporting context.
Return exactly THREE semantically plausible but wrong answer choices. The wrong answer choices should be sufficiently close to the correct 
answer that a knowledgeable test taker might confuse them.  

Rules:
- Make them concise noun phrases (a few words).
- Same answer type/category as the correct answer.
- Must be clearly contradicted by or NOT supported by the context.
- Avoid negations like "not", "none of the above", or humorous/absurd options.
- Do not repeat the correct answer with trivial rephrasing.
- Keep them distinct from each other.

STRICT OUTPUT (JSON only):
{
  "distractors": ["<D1>", "<D2>", "<D3>"]
}
"""


def _ask_gpt5_for_distractors(client, model: str, question: str, correct_answer: str, context: str,
                              temperature: float = 0.5, max_retries: int = 3) -> List[str]:
    user_prompt = (
        f"QUESTION: {question}\n"
        f"CORRECT ANSWER: {correct_answer}\n"
        f"CONTEXT: {context}\n\n"
        "Write three plausible but incorrect options following the JSON schema."
    )
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": DISTRACTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_text = resp.output_text
            json_text = _extract_json_block(raw_text)
            data = json.loads(json_text)

            if not isinstance(data, dict) or "distractors" not in data:
                raise ValueError("Missing 'distractors' key in JSON.")
            if not isinstance(data["distractors"], list) or len(data["distractors"]) < 1:
                raise ValueError("'distractors' must be a non-empty list.")

            return [str(x).strip() for x in data["distractors"]]
        except Exception as e:
            if attempt == max_retries - 1:
                # The original hit an undefined name here (crash). Returning an empty
                # list lets _ensure_unique_distractors pad with fillers instead.
                print(f"[warn] distractor generation failed after {max_retries} attempts: {e}")
                return []
            time.sleep(backoff)
            backoff *= 2.0
    return []


# -------------------- Builder --------------------


def build_items_from_bioasq(client, model: str, split: str = "train", limit: int = 3266,
                            shuffle_seed: int = 2025, temperature: float = 0.6) -> List[Dict[str, Any]]:
    """
    Loads kroshan/BioASQ, extracts (question, correct answer, context),
    asks GPT-5 for 3 distractors, shuffles options, and returns items as:
    {
      "question": <str>,
      "options": [A, B, C, D],
      "correct_index": <int>,
      "meta": {"answer": <gold>, "context": <context>, "id": <optional id if present>}
    }
    """
    from datasets import load_dataset

    ds = load_dataset("kroshan/BioASQ")[split]
    items: List[Dict[str, Any]] = []

    if shuffle_seed is not None:
        random.seed(shuffle_seed)

    n = len(ds) if limit is None else min(limit, len(ds))
    questions = []
    for i in range(n):
        row = ds[len(ds) - 1 - i]
        q: str = row.get("question", "") or ""
        if q not in questions:
            questions.append(q)
        else:
            continue
        text: str = row.get("text", "") or ""

        gold_answer, ctx = parse_answer_and_context(text)
        if not gold_answer or not q:
            continue

        gpt_distractors = _ask_gpt5_for_distractors(
            client, model, question=q, correct_answer=gold_answer, context=ctx, temperature=temperature
        )
        distractors = _ensure_unique_distractors(gpt_distractors, correct=gold_answer)

        labeled = [("correct", gold_answer)] + [("wrong", d) for d in distractors]
        random.shuffle(labeled)

        options = [opt for _, opt in labeled]
        correct_index = next(idx for idx, (tag, _) in enumerate(labeled) if tag == "correct")

        item = {
            "question": q.strip(),
            "options": options,
            "correct_index": correct_index,
            "meta": {
                "answer": gold_answer,
                "context": ctx,
                "id": row.get("id", i)
            }
        }
        print('RESULT', item)
        items.append(item)
        print(i)

    return items


def save_as_jsonl(results: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build BioASQ multiple-choice questions (port of asq_questions.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output", default="data/bioasq_mcq_harder_full.jsonl")
    ap.add_argument("--model", default="gpt-5", help="Distractor-generating model")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=3266)
    ap.add_argument("--shuffle_seed", type=int, default=2025)
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Kept for parity; the Responses API call does not pass it")
    args = ap.parse_args()

    import os
    from openai import OpenAI

    client = OpenAI()
    items = build_items_from_bioasq(
        client, args.model, split=args.split, limit=args.limit,
        shuffle_seed=args.shuffle_seed, temperature=args.temperature,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_as_jsonl(items, args.output)
    print(f"[done] {len(items)} questions -> {args.output}")


if __name__ == "__main__":
    main()
