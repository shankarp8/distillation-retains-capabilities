#!/usr/bin/env python
"""Aggregate run results and select capability-preserving checkpoints.

Walks a directory of runs, reads the metrics written by ``eval_domain.py``
(``{task}_eval/metrics.json``) and ``eval_general.py``
(``general_eval/summary.json``, keyed by lm-eval task name), and produces the
table used in the paper.

Benchmark columns map onto the Open LLM Leaderboard v2 task names that
``eval_general.py`` runs:

    BBH     leaderboard_bbh         acc_norm
    GPQA    leaderboard_gpqa        acc_norm
    MMLU-P  leaderboard_mmlu_pro    acc
    MuSR    leaderboard_musr        acc_norm
    IFEval  ifeval / leaderboard_ifeval   --ifeval_metric (default inst_level_strict_acc)
    Math    leaderboard_math_hard   exact_match
    Code    humaneval_instruct      pass@1

On the IFEval metric: the M_post IFEval values in Tables 2–4 (80.70 / 85.25 /
72.66 / 85.49) line up with ``inst_level_strict_acc`` for those four models,
so that is the default. The grid-scan helper in the original tree
(``scan_grid_pareto.py``) filtered on ``inst_level_loose_acc`` instead; pass
``--ifeval_metric inst_level_loose_acc`` to reproduce that filter.

The selection rule (the ``CP`` superscript in Tables 3 and 6): among all
learning rates for a method, take the checkpoint with the highest adaptation
score subject to no more than a 5-point drop on each of IFEval, MATH, and
HumanEval relative to the initial post-trained model.

Usage
-----
    python analysis/collect_results.py --runs_dir runs \
        --baseline runs/qwen7b_base --task kup --output results/table3.csv

    python analysis/collect_results.py --runs_dir runs --baseline runs/qwen7b_base \
        --task kup --select_cp --max_drop 5.0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd

# Column key -> (candidate lm-eval task names in priority order, candidate metric keys)
BENCHMARKS: Dict[str, tuple] = {
    "bbh": (["leaderboard_bbh"], ["acc_norm,none"]),
    "gpqa": (["leaderboard_gpqa"], ["acc_norm,none"]),
    "mmlu_pro": (["leaderboard_mmlu_pro"], ["acc,none"]),
    "musr": (["leaderboard_musr"], ["acc_norm,none"]),
    # ``ifeval`` (the eval2–eval6 re-runs with max_gen_toks=2048) is preferred
    # over the copy inside the leaderboard group when both are present.
    "ifeval": (["ifeval", "leaderboard_ifeval"], None),   # metric set by --ifeval_metric
    "math": (["leaderboard_math_hard"], ["exact_match,none"]),
    "code": (["humaneval_instruct", "humaneval"], ["pass@1,create_test", "pass@1,none"]),
}

TASK_ORDER = ["bbh", "gpqa", "mmlu_pro", "musr", "ifeval", "math", "code"]
TASK_LABELS = {
    "bbh": "BBH",
    "gpqa": "GPQA",
    "mmlu_pro": "MMLU-P",
    "musr": "MuSR",
    "ifeval": "IFEval",
    "math": "Math",
    "code": "Code",
}

# Post-training skills that define the capability-preserving constraint (Section 6.1).
CP_CONSTRAINT_TASKS = ["ifeval", "math", "code"]


def pick_metric(metrics: Dict, candidates: Optional[List[str]]) -> Optional[float]:
    if candidates:
        for key in candidates:
            if key in metrics:
                return float(metrics[key]) * 100
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and "stderr" not in key and key != "alias":
            return float(value) * 100
    return None


def read_general(summary: Dict, ifeval_metric: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col, (task_names, metric_keys) in BENCHMARKS.items():
        keys = metric_keys
        if col == "ifeval":
            keys = [f"{ifeval_metric},none", ifeval_metric]
        for name in task_names:
            if name in summary:
                value = pick_metric(summary[name], keys)
                if value is not None:
                    out[col] = round(value, 2)
                    break
    return out


def read_run(run_dir: str, task: str, ifeval_metric: str) -> Optional[Dict]:
    """Read one checkpoint directory into a flat record."""
    record: Dict[str, object] = {"run": os.path.basename(run_dir), "path": run_dir}

    # config.json lives in the run dir (parent of step_N); check both.
    for cfg_dir in (run_dir, os.path.dirname(run_dir)):
        config_path = os.path.join(cfg_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            record.update({
                "method": cfg.get("method"),
                "model": cfg.get("model_name"),
                "corpus": cfg.get("corpus"),
                "lr": cfg.get("lr"),
            })
            break

    domain_path = os.path.join(run_dir, f"{task}_eval", "metrics.json")
    if os.path.exists(domain_path):
        with open(domain_path) as f:
            record[task] = round(json.load(f)["accuracy"] * 100, 2)

    general_path = os.path.join(run_dir, "general_eval", "summary.json")
    if os.path.exists(general_path):
        with open(general_path) as f:
            record.update(read_general(json.load(f), ifeval_metric))

    has_results = any(k in record for k in [task] + TASK_ORDER)
    return record if has_results else None


def find_runs(runs_dir: str) -> List[str]:
    """Every directory that looks like a checkpoint with evaluation output."""
    found = []
    for root, dirs, _ in os.walk(runs_dir):
        if os.path.basename(root) in {"general_eval", "kup_eval", "bioasq_eval", "lora_modules"}:
            dirs[:] = []
            continue
        if any(os.path.isdir(os.path.join(root, d)) for d in ("general_eval", "kup_eval", "bioasq_eval")):
            found.append(root)
    return sorted(found)


def select_capability_preserving(df: pd.DataFrame, baseline: pd.Series,
                                 task: str, max_drop: float) -> pd.DataFrame:
    """Best-adaptation checkpoint per (method, model) within the forgetting budget."""
    constraints = [t for t in CP_CONSTRAINT_TASKS if t in df.columns and t in baseline]
    if not constraints:
        print("[warn] no constraint tasks found; returning all runs unfiltered")
        return df

    eligible = df.copy()
    for t in constraints:
        eligible[f"drop_{t}"] = baseline[t] - eligible[t]

    mask = pd.Series(True, index=eligible.index)
    for t in constraints:
        mask &= eligible[f"drop_{t}"] <= max_drop
    eligible = eligible[mask]

    if eligible.empty:
        print(f"[warn] no checkpoint satisfies the <={max_drop} point constraint")
        return eligible

    group_cols = [c for c in ("method", "model") if c in eligible.columns]
    if not group_cols:
        return eligible.nlargest(1, task)
    return eligible.loc[eligible.groupby(group_cols)[task].idxmax()].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate results across runs")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--task", default="kup", choices=["kup", "bioasq"])
    ap.add_argument("--baseline", default=None,
                    help="Path to the evaluated initial model (M_post), for drop columns")
    ap.add_argument("--select_cp", action="store_true",
                    help="Keep only capability-preserving checkpoints")
    ap.add_argument("--max_drop", type=float, default=5.0)
    ap.add_argument("--ifeval_metric", default="inst_level_strict_acc",
                    choices=["prompt_level_strict_acc", "inst_level_strict_acc",
                             "prompt_level_loose_acc", "inst_level_loose_acc"])
    ap.add_argument("--output", default=None, help="Write the table to CSV")
    args = ap.parse_args()

    records = [r for d in find_runs(args.runs_dir) if (r := read_run(d, args.task, args.ifeval_metric))]
    if not records:
        raise SystemExit(f"No evaluated runs found under {args.runs_dir}")

    df = pd.DataFrame(records)

    baseline = None
    if args.baseline:
        baseline_record = read_run(args.baseline, args.task, args.ifeval_metric)
        if baseline_record:
            baseline = pd.Series(baseline_record)
            df = df[df["path"] != args.baseline].reset_index(drop=True)

    if args.select_cp:
        if baseline is None:
            raise SystemExit("--select_cp requires --baseline (M_post reference scores)")
        df = select_capability_preserving(df, baseline, args.task, args.max_drop)

    columns = [c for c in ("method", "model", "lr", "run") if c in df.columns]
    columns += [t for t in TASK_ORDER if t in df.columns] + [args.task]
    columns = list(dict.fromkeys(c for c in columns if c in df.columns))
    df = df[columns].sort_values([c for c in ("method", "lr") if c in df.columns])

    display = df.rename(columns={**TASK_LABELS, args.task: args.task.upper()})
    column_order = list(display.columns)

    if baseline is not None:
        row = {"method": "M_post"}
        row.update({TASK_LABELS.get(t, t): baseline.get(t) for t in TASK_ORDER if t in baseline})
        row[args.task.upper()] = baseline.get(args.task)
        display = pd.concat([pd.DataFrame([row]), display], ignore_index=True)
        display = display.reindex(columns=column_order)

    print(display.to_string(index=False))

    if baseline is not None:
        drops = pd.DataFrame({
            "method": display["method"],
            **{
                TASK_LABELS[t]: (baseline[t] - display[TASK_LABELS[t]]).round(2)
                for t in CP_CONSTRAINT_TASKS
                if t in baseline and TASK_LABELS[t] in display
            },
        })
        print("\nDrop vs. M_post on post-training skills (positive = forgetting):")
        print(drops.to_string(index=False))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        display.to_csv(args.output, index=False)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
