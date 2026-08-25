#!/usr/bin/env python
"""General capability evaluation (T_gen) via lm-evaluation-harness.

Reproduces the lm-eval invocation used by the original wrapper scripts
(eval2–eval10, eval_args, humaneval_args, qwen3_eval). The task set is the
Open LLM Leaderboard v2 group — ``leaderboard`` expands to leaderboard_bbh,
leaderboard_gpqa, leaderboard_ifeval, leaderboard_math_hard,
leaderboard_mmlu_pro, leaderboard_musr — plus ``humaneval_instruct``. These are
NOT the same task definitions as the bare ``bbh`` / ``mmlu_pro`` / ``musr``
names in lm-eval (different prompting and scoring), so the leaderboard names
are the default and should not be swapped.

The exact command that is reproduced:

    python -m lm_eval --model vllm \
        --model_args pretrained=<ckpt>,dtype=bfloat16,gpu_memory_utilization=0.6 \
        --tasks leaderboard,humaneval_instruct \
        --apply_chat_template --fewshot_as_multiturn \
        --batch_size 512 --output_path <dir> \
        --log_samples --confirm_run_unsafe_code --trust_remote_code

Usage
-----
    python scripts/eval_general.py --model_path runs/qwen7b_kup_disc/step_5000

    # IFEval + MATH-Hard re-run exactly as eval2–eval6 did it
    # (max_gen_toks=2048, batch 4096):
    python scripts/eval_general.py --model_path ... --preset ifeval_math

    # Qwen3: thinking on for everything except BBH and MMLU-Pro, with the
    # sampling settings from eval9/eval10 for the thinking group.
    python scripts/eval_general.py --model_path ... --qwen3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

DEFAULT_TASKS = "leaderboard,humaneval_instruct"

# Members of the ``leaderboard`` group, used when a run has to be split
# (Qwen3 thinking vs. non-thinking).
LEADERBOARD_SUBTASKS = [
    "leaderboard_bbh",
    "leaderboard_gpqa",
    "leaderboard_ifeval",
    "leaderboard_math_hard",
    "leaderboard_mmlu_pro",
    "leaderboard_musr",
]

CODE_TASKS = {"humaneval", "humaneval_instruct"}

# Original model_args (all wrappers).
BASE_MODEL_ARGS = "dtype=bfloat16,gpu_memory_utilization=0.6"

# Qwen3 (eval7–eval10): thinking on, strip the think block, Qwen3's
# recommended sampling for the thinking group.
QWEN3_THINK_MODEL_ARGS = "enable_thinking=True,think_end_token='</think>'"
QWEN3_THINK_GEN_KWARGS = "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0"
# Tasks the paper evaluates with thinking disabled for Qwen3 (Appendix C).
QWEN3_NO_THINK_TASKS = {"leaderboard_bbh", "leaderboard_mmlu_pro"}
QWEN3_NO_THINK_MODEL_ARGS = "enable_thinking=False"

PRESETS = {
    # eval2.py … eval6.py
    "ifeval_math": {"tasks": "ifeval,leaderboard_math_hard", "max_gen_toks": 2048, "batch_size": "4096"},
    # eval9.py / eval10.py
    "leaderboard": {"tasks": DEFAULT_TASKS, "max_gen_toks": None, "batch_size": "512"},
    # eval_args.py / eval13.py
    "ifeval": {"tasks": "ifeval", "max_gen_toks": None, "batch_size": "auto"},
    # humaneval_args.py
    "humaneval": {"tasks": "humaneval_instruct", "max_gen_toks": None, "batch_size": "auto"},
}


def expand_tasks(task_str: str) -> list[str]:
    tasks = []
    for t in task_str.split(","):
        t = t.strip()
        if not t:
            continue
        if t == "leaderboard":
            tasks.extend(LEADERBOARD_SUBTASKS)
        else:
            tasks.append(t)
    return tasks


def build_command(args, tasks: list[str], model_args_extra: str, gen_kwargs: str | None,
                  output_path: str) -> list[str]:
    model_args = f"pretrained={args.model_path}"
    if model_args_extra:
        model_args += f",{model_args_extra}"
    model_args += f",{BASE_MODEL_ARGS}"
    if args.max_gen_toks:
        model_args += f",max_gen_toks={args.max_gen_toks}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
    ]
    if gen_kwargs:
        cmd += ["--gen_kwargs", gen_kwargs]
    cmd += [
        "--tasks", ",".join(tasks),
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        "--batch_size", args.batch_size,
        "--output_path", output_path,
        "--log_samples",
        "--confirm_run_unsafe_code",
        "--trust_remote_code",
    ]
    return cmd


def run(cmd: list[str], env: dict, log_prefix: str) -> int:
    print("[lm-eval] " + " ".join(cmd))
    with open(f"{log_prefix}_stdout.txt", "w") as out, open(f"{log_prefix}_stderr.txt", "w") as err:
        proc = subprocess.run(cmd, env=env, stdout=out, stderr=err)
    if proc.returncode != 0:
        print(f"[lm-eval] FAILED (exit {proc.returncode}); see {log_prefix}_stderr.txt")
    return proc.returncode


def collect_results(output_path: str) -> dict:
    """lm-eval writes results_<timestamp>.json under output_path/<model>/; merge them."""
    merged = {}
    for path in sorted(glob.glob(os.path.join(output_path, "**", "results_*.json"), recursive=True)):
        with open(path) as f:
            payload = json.load(f)
        merged.update(payload.get("results", {}))
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(
        description="General capability evaluation (lm-eval, leaderboard tasks)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--preset", default="leaderboard", choices=sorted(PRESETS),
                    help="Which original wrapper to reproduce (sets tasks / max_gen_toks / batch_size)")
    ap.add_argument("--tasks", default=None, help="Override the preset's task string")
    ap.add_argument("--max_gen_toks", type=int, default=None, help="Override the preset's max_gen_toks")
    ap.add_argument("--batch_size", default=None, help="Override the preset's batch size")
    ap.add_argument("--output_dir", default=None, help="Default: {model_path}/general_eval")
    ap.add_argument("--qwen3", action="store_true",
                    help="Qwen3: thinking on (+ eval9/10 sampling) for all tasks except BBH and "
                         "MMLU-Pro, which run with enable_thinking=False")
    ap.add_argument("--cuda_visible_devices", default=None,
                    help="Set CUDA_VISIBLE_DEVICES for the lm-eval subprocess (originals used '0')")
    args = ap.parse_args()

    preset = PRESETS[args.preset]
    task_str = args.tasks or preset["tasks"]
    args.max_gen_toks = args.max_gen_toks if args.max_gen_toks is not None else preset["max_gen_toks"]
    args.batch_size = args.batch_size or preset["batch_size"]

    args.model_path = os.path.abspath(args.model_path)
    if not os.path.isdir(args.model_path):
        raise SystemExit(f"Model path does not exist: {args.model_path}")

    args.output_dir = args.output_dir or os.path.join(args.model_path, "general_eval")
    os.makedirs(args.output_dir, exist_ok=True)

    env = os.environ.copy()
    for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(var, None)
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["HF_ALLOW_CODE_EVAL"] = "1"

    # (tasks, extra model_args, gen_kwargs, run name)
    if args.qwen3:
        # The group has to be split, so expand ``leaderboard`` into its members.
        requested = expand_tasks(task_str)
        think = [t for t in requested if t not in QWEN3_NO_THINK_TASKS]
        nothink = [t for t in requested if t in QWEN3_NO_THINK_TASKS]
        groups = [
            (think, QWEN3_THINK_MODEL_ARGS, QWEN3_THINK_GEN_KWARGS, "think"),
            (nothink, QWEN3_NO_THINK_MODEL_ARGS, None, "nothink"),
        ]
    else:
        # Pass the task string through untouched (``leaderboard`` stays a group,
        # exactly as the original wrappers invoked it).
        groups = [([t.strip() for t in task_str.split(",") if t.strip()], "", None, "main")]

    results = {}
    for tasks, extra, gen_kwargs, name in groups:
        if not tasks:
            continue
        output_path = os.path.join(args.output_dir, f"lm_eval_{name}")
        cmd = build_command(args, tasks, extra, gen_kwargs, output_path)
        prefix = os.path.join(args.output_dir, name)
        if run(cmd, env, prefix) == 0:
            results.update(collect_results(output_path))

    if results:
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[done] {len(results)} task results -> {summary_path}")
        for task, metrics in sorted(results.items()):
            shown = {k: round(v * 100, 2) for k, v in metrics.items()
                     if isinstance(v, float) and "stderr" not in k}
            print(f"  {task:28s} {shown}")


if __name__ == "__main__":
    main()
