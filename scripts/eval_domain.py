#!/usr/bin/env python
"""Knowledge-adaptation evaluation (T_D): KUP and BioASQ multiple choice.

Both tasks are direct ports of the scripts that produced the paper's numbers.
Prompts, sampling parameters, answer parsers, and vote/tie handling are kept
verbatim; only the model path, data path, and output directory are arguments.

  --task kup     ports ``new_kup_eval4.py``
                 (raw prompt, no chat template, n=5 @ T=1.0, pandas-mode vote)
  --task bioasq  ports ``eval_vllm.py``
                 (system prompt + chat template, seeded per-question option
                 shuffle, question dedup, n=5 @ T=1.0 top_p=0.9, tie -> None)

Usage
-----
    python scripts/eval_domain.py --task kup \
        --model_path runs/qwen7b_kup_disc/step_5000 --data data/wikimcq_df.pickle

    python scripts/eval_domain.py --task bioasq \
        --model_path runs/qwen7b_bioasq_disc/step_3266 \
        --data data/bioasq_mcq_harder_full.jsonl

Note on the two tasks not sharing a pipeline: they never did. The KUP script
sends a raw string prompt with a fixed "knowledge cutoff" preamble and votes
with ``Series.mode()``; the BioASQ script wraps a system + user message in the
model's chat template, shuffles options per question with a seeded RNG, and
treats vote ties as unanswered. Unifying them would change the numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# ==========================================================================
# KUP  (port of new_kup_eval4.py)
# ==========================================================================

# NOTE: the "Knowledge Cutoff Date" preamble is part of the prompt that every
# KUP evaluation in the original tree used. Appendix D.1 of the paper omits it.
KUP_TEMPLATE = """Your Knowledge Cutoff Date: December 2026.

You will answer a multiple-choice question about {entity}.
- Put your reasoning in: <think> ... </think>
- Put ONLY the final answer letter (A/B/C/D) in: <answer> ... </answer>
- Do not include any other text outside these tags.

{question}
"""

# Entities with malformed questions, excluded from scoring.
KUP_EXCLUDED_ENTITY_IDS = [4641, 7759, 1868, 234]

KUP_N = 5
KUP_MAX_TOKENS = 16000          # vLLM caps generation at max_model_len - prompt_len
KUP_TEMPERATURE = 1.0
KUP_DTYPE = "float16"
KUP_GPU_MEMORY_UTILIZATION = 0.85
KUP_MAX_MODEL_LEN = 2048

ANSWER_TAG_RE = re.compile(
    r"<answer>\s*([ABCD])\s*</answer>",
    flags=re.IGNORECASE | re.DOTALL,
)

THINK_TAG_RE = re.compile(
    r"<think>.*?</think>",
    flags=re.IGNORECASE | re.DOTALL,
)


def parse_answer_from_tags(response: str) -> str | None:
    """
    Preferred parser: extract a single multiple-choice letter from <answer>...</answer>.
    If multiple <answer> tags exist, returns the LAST one (often the final answer).
    """
    if not isinstance(response, str) or not response.strip():
        return None
    matches = ANSWER_TAG_RE.findall(response)
    if not matches:
        return None
    return matches[-1].upper()


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks (useful for fallback parsing)."""
    if not isinstance(text, str):
        return ""
    return THINK_TAG_RE.sub("", text)


def parse_answer_after_answer_tag(response: str) -> str | None:
    lines = response.split('\n')
    for i, line in enumerate(lines):
        if "answer:" in line.lower():
            inline = re.search(r'answer\s*[:\-]?\s*([abcd])\b', line.lower())
            if inline:
                return inline.group(1).upper()

            if i + 1 < len(lines):
                next_line = lines[i + 1].strip().lower()
                match = re.match(r'^([abcd])[\:\.\)\-]', next_line)
                if match:
                    return match.group(1).upper()
    return None


def parse_answer_literal(response: str) -> str | None:
    lines = response.strip().split('\n')
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if "Answer:" in line_stripped:
            parts = line_stripped.split("Answer:")
            if len(parts) > 1:
                after = parts[1].strip()
                if after and after[0] in {'A', 'B', 'C', 'D'}:
                    return after[0]

            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and next_line[0] in {'A', 'B', 'C', 'D'}:
                    if len(next_line) == 1 or next_line[1] in {':', '.', ' '}:
                        return next_line[0]
            return None
    return None


def parse_answer_first_nonempty_line_heuristic(response: str) -> str | None:
    """
    First non-empty line heuristic, kept as a fallback of last resort.
    """
    try:
        ans = [line for line in response.split('\n') if line.strip()][0]
        if "Answer" in ans:
            ans = ans.split("Answer")[1][:5]
        ans = ans.split('.')[0].replace(':', '').lower().strip()

        if 'a' in ans and all(x not in ans for x in ('b', 'c', 'd')):
            return 'A'
        if 'b' in ans and all(x not in ans for x in ('a', 'c', 'd')):
            return 'B'
        if 'c' in ans and all(x not in ans for x in ('a', 'b', 'd')):
            return 'C'
        if 'd' in ans and all(x not in ans for x in ('a', 'b', 'c')):
            return 'D'
        return None
    except Exception:
        return None


def parse_answer_with_fallback(response: str) -> tuple[str | None, str]:
    """
    Returns (answer, parser_used).
    Priority:
      1) <answer> tag
      2) literal "Answer:" parser
      3) "answer:" tag-ish parser
      4) first-nonempty-line heuristic
    <think> blocks are stripped before fallback parsing to reduce noise.
    """
    a = parse_answer_from_tags(response)
    if a is not None:
        return a, "tags"

    cleaned = strip_think_blocks(response)

    a = parse_answer_literal(cleaned)
    if a is not None:
        return a, "literal"

    a = parse_answer_after_answer_tag(cleaned)
    if a is not None:
        return a, "after_answer_tag"

    a = parse_answer_first_nonempty_line_heuristic(cleaned)
    if a is not None:
        return a, "heuristic_first_line"

    return None, "none"


def run_kup(args: argparse.Namespace) -> dict:
    from vllm import LLM, SamplingParams

    MODEL_PATH = args.model_path
    DATA_PATH = args.data
    OUTPUT_DIR = args.output_dir or os.path.join(MODEL_PATH, "kup_eval")
    N = args.n
    MAX_TOKENS = args.max_tokens
    TEMPERATURE = args.temperature
    TEMPLATE = KUP_TEMPLATE

    print('Evaluating model saved at path {}'.format(MODEL_PATH))

    llm_kwargs = dict(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        revision="main",
    )
    if args.download_dir:
        llm_kwargs["download_dir"] = args.download_dir
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        n=N,
    )

    df = pd.read_pickle(DATA_PATH).dropna().reset_index(drop=True)
    df = df[~df['entity_id'].isin(KUP_EXCLUDED_ENTITY_IDS)].reset_index(drop=True)

    input_prompts = [
        TEMPLATE.format(entity=row["entity"], question=row["question"])
        for _, row in df.iterrows()
    ]

    def complete(prompts, n=N):
        """
        Run generation and return a long-format DataFrame with one row per
        (prompt, sample) pair, preserving the prompt and a sample index.
        """
        print(f"Generating with vLLM for {len(prompts)} prompts (n={n}) ...")
        outputs = llm.generate(prompts, sampling_params)

        rows = []
        for prompt_idx, out in enumerate(outputs):
            prompt_text = out.prompt  # the exact prompt vLLM saw
            for sample_idx, o in enumerate(out.outputs):
                rows.append({
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt_text,
                    "response": o.text,
                })
        return pd.DataFrame(rows)

    def run_eval():
        fact_response_df = complete(input_prompts)
        # Attach per-prompt metadata. Each prompt produces N rows in order.
        fact_response_df["entity_id"] = df["entity_id"].repeat(N).values
        fact_response_df["entity"] = df["entity"].repeat(N).values
        fact_response_df["question"] = df["question"].repeat(N).values
        return fact_response_df

    def parse_response(fact_response_df):
        # Preferred: tag-based. Fallback: the older parsers.
        parsed = fact_response_df["response"].apply(parse_answer_with_fallback)
        fact_response_df["model_answer"] = parsed.apply(lambda x: x[0])
        fact_response_df["parse_method"] = parsed.apply(lambda x: x[1])

        majority_vote = (
            fact_response_df.groupby('entity_id')['model_answer']
            .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            .reset_index()
        )
        return fact_response_df, majority_vote

    fact_response_df = run_eval()
    fact_response_df['answer'] = df['answer'].repeat(N).values
    fact_response_df, majority_vote = parse_response(fact_response_df)

    fact_response_df['correct'] = fact_response_df['model_answer'] == fact_response_df['answer']
    confidence_df = (
        fact_response_df.groupby('entity_id')['correct']
        .mean()
        .reset_index()
        .rename(columns={"correct": "confidence"})
    )
    summary_df = df.merge(majority_vote, on="entity_id").merge(confidence_df, on="entity_id")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fact_response_df.to_pickle(os.path.join(OUTPUT_DIR, "raw_responses.pickle"))
    summary_df.to_pickle(os.path.join(OUTPUT_DIR, "summary_results.pickle"))

    # ---- Save all generations (with prompts) to JSON ----
    generations_records = []
    for row in fact_response_df.itertuples(index=False):
        rec = {
            "entity_id": (int(row.entity_id)
                          if pd.notna(row.entity_id) and not isinstance(row.entity_id, str)
                          else row.entity_id),
            "entity": row.entity,
            "question": row.question,
            "prompt": row.prompt,
            "sample_idx": int(row.sample_idx),
            "response": row.response,
            "model_answer": row.model_answer,
            "parse_method": row.parse_method,
            "gold_answer": row.answer,
            "correct": bool(row.correct) if pd.notna(row.correct) else None,
        }
        generations_records.append(rec)

    generations_payload = {
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "n_samples_per_prompt": N,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "template": TEMPLATE,
        "generations": generations_records,
    }

    json_path = os.path.join(OUTPUT_DIR, "generations.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(generations_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(generations_records)} generations to {json_path}")

    parse_stats = fact_response_df["parse_method"].value_counts(dropna=False)
    print("Parse method counts:\n", parse_stats.to_string())

    # Same accuracy computation as analyze_kup_results.py / the tail of new_kup_eval4.py.
    summary_df['correct'] = (summary_df['answer'] == summary_df['model_answer'])
    accuracy = float(summary_df['correct'].mean())
    unparsed = int(summary_df['model_answer'].isna().sum())
    print(accuracy)
    print(summary_df['correct'].isna().sum())
    print(unparsed)

    metrics = {
        "task": "kup",
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "accuracy": accuracy,
        "n_questions": int(len(summary_df)),
        "unparsed": unparsed,
        "n_samples": N,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Done. Saved outputs to", OUTPUT_DIR)
    print('-' * 100)
    return metrics


# ==========================================================================
# BioASQ  (port of eval_vllm.py)
# ==========================================================================

LETTER_TO_NUM = {chr(ord('A') + i): i + 1 for i in range(26)}
WORDNUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

BIOASQ_ANSWER_TAG_RE = re.compile(r"(?is)<answer>\s*([^<]+?)\s*</answer>")
NUMERAL_RE = re.compile(r"\b([1-9][0-9]*)\b")
LETTER_RE = re.compile(r"\b([A-Z])\b", re.I)

BIOASQ_SEED = 123
BIOASQ_K = 5
BIOASQ_MAX_NEW_TOKENS = 1024
BIOASQ_TEMPERATURE = 1.0
BIOASQ_TOP_P = 0.9
BIOASQ_GPU_MEMORY_UTILIZATION = 0.9
BIOASQ_STOP = ["\n\nQuestion:", "\n\n<|assistant|>", "\n\n<|user|>"]

SYSTEM_INSTRUCTIONS = (
    "You are a careful question-answering assistant. You may privately think inside <think>...</think>, "
    "but you must provide your final choice strictly as the numeral of the correct option (1, 2, 3, 4) inside <answer>...</answer>. "
    "Do not include any other content inside <answer> tags."
)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def word_to_int(token: str) -> Optional[int]:
    t = token.strip().lower()
    if t in WORDNUMS:
        return WORDNUMS[t]
    return None


def clamp_choice(n: int, k: int) -> Optional[int]:
    # Kept as in the original: no clamping is applied.
    return n


def parse_answer(text: str, k: int) -> Optional[int]:
    """
    Parse the model output to a 1..k integer.
    Priority: <answer>...</answer> → numerals/letters/words.
    Fallbacks scan entire text.
    """
    if not text:
        return None

    # 1) Look inside <answer>...</answer>
    m = BIOASQ_ANSWER_TAG_RE.search(text)
    if m:
        inner = m.group(1).strip()
        # Try numeral inside answer tag
        nmatch = NUMERAL_RE.search(inner)
        if nmatch:
            n = int(nmatch.group(1))
            return clamp_choice(n, k)
        # Try letter A.., mapping to 1..k
        lmatch = LETTER_RE.search(inner)
        if lmatch:
            n = LETTER_TO_NUM.get(lmatch.group(1).upper())
            return clamp_choice(n, k)
        # Try word numbers
        w = word_to_int(inner)
        if w is not None:
            return clamp_choice(w, k)

    # 2) Fallback: look for first plausible numeral anywhere
    nmatch = NUMERAL_RE.search(text)
    if nmatch:
        n = int(nmatch.group(1))
        n = clamp_choice(n, k)
        if n is not None:
            return n

    # 3) Fallback: look for letter A.., choose first that maps in range
    for lmatch in LETTER_RE.finditer(text):
        n = LETTER_TO_NUM.get(lmatch.group(1).upper())
        if clamp_choice(n, k):
            return n

    # 4) Fallback: word number
    tokens = re.findall(r"[A-Za-z]+", text)
    for tok in tokens:
        w = word_to_int(tok)
        if clamp_choice(w, k):
            return w

    return None


def majority_vote(choices: List[Optional[int]]) -> Tuple[Optional[int], bool]:
    """Returns (winner, tie_flag). Ignores None. If tie, returns None and tie_flag=True."""
    valid = [c for c in choices if c is not None]
    if not valid:
        return None, False
    counts = Counter(valid)
    if len(counts) == 1:
        return valid[0], False
    top = counts.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return None, True
    return top[0][0], False


def build_prompt_body(question: str, options: List[str]) -> str:
    opts_str = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))
    return (
        "Answer the multiple-choice question by reasoning briefly in <think>...</think> and then giving ONLY the numeral of the correct option inside <answer>...</answer>.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts_str}\n\n"
        "Rules:\n"
        " - Put your final choice as a numeral only (e.g., 2) inside <answer> tags.\n"
        " - Do NOT repeat the option text inside <answer>.\n"
        " - Choose exactly one option.\n"
    )


def apply_chat_template(tokenizer, system: str, user: str) -> str:
    """
    If the tokenizer provides a chat template, use it; otherwise, return a plain text prompt.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback plain prompt
    return f"{system}\n\n{user}\n\nAnswer:\n"


def run_bioasq(args: argparse.Namespace) -> dict:
    from vllm import LLM, SamplingParams

    data_path = Path(args.data)
    model_name = args.model_path
    batch_seed = args.seed
    k_generations = args.n
    output_dir = args.output_dir or os.path.join(model_name, "bioasq_eval")

    # Load model with vLLM (dtype is left to vLLM's default, as in the original)
    print(f"Loading model {model_name} with vLLM...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        n=k_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=list(BIOASQ_STOP),
    )

    # Load all examples first (dedup by question text; ex_idx keeps the raw file index,
    # which seeds the per-question option shuffle)
    print("Loading examples...")
    examples = []
    unique_qs = set()
    for ex_idx, ex in enumerate(load_jsonl(data_path)):
        q = ex["question"]
        if q in unique_qs:
            continue
        unique_qs.add(q)
        examples.append((ex_idx, ex))

    print(f"Loaded {len(examples)} unique examples")

    prompts = []
    metadata = []  # (shuffled_correct_idx, num_options, ex_idx, question, shuffled_options)

    for ex_idx, ex in examples:
        q = ex["question"]
        options = list(ex["options"])
        correct_index = int(ex["correct_index"])  # 0-based

        # Shuffle options deterministically per example index and seed
        local_rng = random.Random((batch_seed + 9973) ^ (ex_idx * 7919))
        perm = list(range(len(options)))
        local_rng.shuffle(perm)
        shuffled_options = [options[i] for i in perm]

        # Map original correct_index to shuffled index
        shuffled_correct_idx = perm.index(correct_index)  # 0-based

        prompt_body = build_prompt_body(q, shuffled_options)
        prompt = apply_chat_template(tokenizer, SYSTEM_INSTRUCTIONS, prompt_body)

        prompts.append(prompt)
        metadata.append((shuffled_correct_idx, len(shuffled_options), ex_idx, q, shuffled_options))

    print("Generating outputs with vLLM...")
    all_outputs = llm.generate(prompts, sampling_params)

    n_total = 0
    n_correct = 0
    n_covered = 0
    n_ties = 0
    vote_hist_by_k = defaultdict(Counter)
    records = []

    for i, output in enumerate(all_outputs):
        shuffled_correct_idx, num_options, ex_idx, q, shuffled_options = metadata[i]

        raw_outputs = [o.text for o in output.outputs]

        parsed_choices = []
        for out in raw_outputs:
            choice = parse_answer(out, k=num_options)  # 1..k or None
            parsed_choices.append(choice)

        # Majority vote (over valid parses only)
        voted, tie = majority_vote(parsed_choices)

        n_total += 1
        if tie:
            n_ties += 1

        if any(c is not None for c in parsed_choices):
            n_covered += 1

        is_correct = False
        if voted is not None:
            pred_shuf_idx_0based = voted - 1
            is_correct = (pred_shuf_idx_0based == shuffled_correct_idx)
            n_correct += int(is_correct)
            vote_hist_by_k[num_options][voted] += 1

        records.append({
            "ex_idx": ex_idx,
            "question": q,
            "options_shuffled": shuffled_options,
            "gold_index_1based": shuffled_correct_idx + 1,
            "prompt": prompts[i],
            "responses": raw_outputs,
            "parsed_choices": parsed_choices,
            "voted": voted,
            "tie": tie,
            "correct": bool(is_correct),
        })

        if n_total % 100 == 0:
            acc = n_correct / n_total
            cov = n_covered / n_total
            print(f"[{n_total} ex] Acc={acc:.3f} | Coverage={cov:.3f} | TieRate={n_ties/n_total:.3f}", flush=True)

    acc = n_correct / n_total if n_total else 0.0
    coverage = n_covered / n_total if n_total else 0.0
    tie_rate = n_ties / n_total if n_total else 0.0
    print("\n=== Evaluation Report ===")
    print(f"Examples:      {n_total}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Coverage:      {coverage:.4f}  (fraction with at least one valid parsed choice)")
    print(f"Tie rate:      {tie_rate:.4f}  (majority vote ties)")
    print(f"Generations/Q: {k_generations}")
    print()

    for K, hist in sorted(vote_hist_by_k.items()):
        total_votes = sum(hist.values())
        if total_votes == 0:
            continue
        print(f"K={K} voted numeral distribution:")
        for n in range(1, K + 1):
            c = hist[n]
            pct = 100.0 * c / total_votes
            print(f"  {n}: {c} ({pct:.1f}%)")
        print()

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "generations.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = {
        "task": "bioasq",
        "model_path": model_name,
        "data_path": str(data_path),
        "accuracy": acc,
        "coverage": coverage,
        "tie_rate": tie_rate,
        "n_questions": n_total,
        "n_samples": k_generations,
        "seed": batch_seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved -> {output_dir}")
    return metrics


# ==========================================================================
# CLI
# ==========================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Domain adaptation evaluation (KUP / BioASQ)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--task", required=True, choices=["kup", "bioasq"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True,
                    help="wikimcq_df.pickle (KUP) or bioasq_mcq_harder_full.jsonl (BioASQ)")
    ap.add_argument("--output_dir", default=None, help="Default: {model_path}/{task}_eval")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)

    # Per-task defaults are resolved after parsing so each task keeps the exact
    # values its original script used.
    ap.add_argument("--n", type=int, default=None, help="Samples per question (both tasks: 5)")
    ap.add_argument("--temperature", type=float, default=None, help="(both tasks: 1.0)")
    ap.add_argument("--max_tokens", type=int, default=None, help="(kup: 16000, bioasq: 1024)")
    ap.add_argument("--gpu_memory_utilization", type=float, default=None,
                    help="(kup: 0.85, bioasq: 0.9)")

    kup = ap.add_argument_group("kup")
    kup.add_argument("--dtype", default=KUP_DTYPE, help="vLLM dtype for KUP")
    kup.add_argument("--max_model_len", type=int, default=KUP_MAX_MODEL_LEN)
    kup.add_argument("--download_dir", default=None, help="Optional vLLM download_dir")

    bio = ap.add_argument_group("bioasq")
    bio.add_argument("--seed", type=int, default=BIOASQ_SEED, help="Option-shuffle seed")
    bio.add_argument("--top_p", type=float, default=BIOASQ_TOP_P)

    args = ap.parse_args()

    if args.task == "kup":
        args.n = KUP_N if args.n is None else args.n
        args.temperature = KUP_TEMPERATURE if args.temperature is None else args.temperature
        args.max_tokens = KUP_MAX_TOKENS if args.max_tokens is None else args.max_tokens
        args.gpu_memory_utilization = (KUP_GPU_MEMORY_UTILIZATION
                                       if args.gpu_memory_utilization is None
                                       else args.gpu_memory_utilization)
        run_kup(args)
    else:
        args.n = BIOASQ_K if args.n is None else args.n
        args.temperature = BIOASQ_TEMPERATURE if args.temperature is None else args.temperature
        args.max_tokens = BIOASQ_MAX_NEW_TOKENS if args.max_tokens is None else args.max_tokens
        args.gpu_memory_utilization = (BIOASQ_GPU_MEMORY_UTILIZATION
                                       if args.gpu_memory_utilization is None
                                       else args.gpu_memory_utilization)
        run_bioasq(args)


if __name__ == "__main__":
    main()
