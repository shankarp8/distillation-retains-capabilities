#!/usr/bin/env python
"""General capability evaluation (T_gen) via lm-evaluation-harness.

Replaces the per-benchmark wrapper scripts (ifeval / humaneval / musr /
mmlu_pro / gsm8k / ...), which differed only in the ``--tasks`` string and a
few harness flags.

Usage
-----
    python scripts/eval_general.py --model_path runs/qwen7b_kup_disc/step_5000
    python scripts/eval_general.py --model_path ... --tasks ifeval,humaneval_instruct
    python scripts/eval_general.py --model_path ... --qwen3_no_think

Notes
-----
* Everything is evaluated with the model's native chat template and greedy
  decoding, with few-shot examples cast into a multi-turn format.
* HumanEval needs ``--confirm_run_unsafe_code``; it is passed automatically
  when a code task is requested.
* For Qwen3, BBH and MMLU-Pro are run with thinking mode disabled — enabling
  it collapses performance on these two tasks (MMLU-Pro drops from ~34.7 to
  ~11.3), so they are run separately from the rest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# The seven benchmarks reported in the paper.
DEFAULT_TASKS = [
    "bbh",
    "gpqa",
    "mmlu_pro",
    "musr",
    "ifeval",
    "math_hard",
    "humaneval_instruct",
]

CODE_TASKS = {"humaneval", "humaneval_instruct"}

# Tasks that must be run with Qwen3 thinking mode off.
QWEN3_NO_THINK_TASKS = {"bbh", "mmlu_pro"}


def build_command(args, tasks: list[str], no_think: bool) -> tuple[list[str], str]:
    model_args = f"pretrained={args.model_path},dtype={args.dtype}"
    if args.gpu_memory_utilization:
        model_args += f",gpu_memory_utilization={args.gpu_memory_utilization}"
    if no_think:
        model_args += ",enable_thinking=False"

    suffix = "_nothink" if no_think else ""
    output_path = os.path.join(args.output_dir, f"eval_results{suffix}.json")

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", args.backend,
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--batch_size", args.batch_size,
        "--output_path", output_path,
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        "--trust_remote_code",
    ]
    if any(t in CODE_TASKS for t in tasks):
        cmd.append("--confirm_run_unsafe_code")
    return cmd, output_path


def run(cmd: list[str], env: dict, log_prefix: str) -> int:
    print("[lm-eval] " + " ".join(cmd))
    with open(f"{log_prefix}_stdout.txt", "w") as out, open(f"{log_prefix}_stderr.txt", "w") as err:
        proc = subprocess.run(cmd, env=env, stdout=out, stderr=err)
    if proc.returncode != 0:
        print(f"[lm-eval] FAILED (exit {proc.returncode}); see {log_prefix}_stderr.txt")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="General capability evaluation")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--output_dir", default=None, help="Default: {model_path}/general_eval")
    ap.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--gpu_memory_utilization", type=float, default=None)
    ap.add_argument("--qwen3_no_think", action="store_true",
                    help="Run BBH and MMLU-Pro with thinking mode disabled (Qwen3)")
    args = ap.parse_args()

    args.model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(args.model_path):
        raise SystemExit(f"Model path does not exist: {args.model_path}")

    args.output_dir = args.output_dir or os.path.join(args.model_path, "general_eval")
    os.makedirs(args.output_dir, exist_ok=True)

    env = os.environ.copy()
    env["HF_ALLOW_CODE_EVAL"] = "1"
    # Strip any inherited distributed-launch variables; lm-eval must run single-process.
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(var, None)

    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.qwen3_no_think:
        groups = [
            ([t for t in requested if t not in QWEN3_NO_THINK_TASKS], False),
            ([t for t in requested if t in QWEN3_NO_THINK_TASKS], True),
        ]
    else:
        groups = [(requested, False)]

    results = {}
    for tasks, no_think in groups:
        if not tasks:
            continue
        cmd, output_path = build_command(args, tasks, no_think)
        prefix = os.path.join(args.output_dir, "nothink" if no_think else "main")
        if run(cmd, env, prefix) == 0 and os.path.exists(output_path):
            with open(output_path) as f:
                payload = json.load(f)
            results.update(payload.get("results", {}))

    if results:
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[done] {len(results)} task results -> {summary_path}")
        for task, metrics in sorted(results.items()):
            primary = next(
                (v for k, v in metrics.items() if isinstance(v, float) and "stderr" not in k),
                None,
            )
            if primary is not None:
                print(f"  {task:24s} {primary * 100:.2f}")


if __name__ == "__main__":
    main()
