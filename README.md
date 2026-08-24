# DiSC: Distillation via Split Contexts

Code for *Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities.*

DiSC adapts a post-trained LM to a new document corpus while preserving the
capabilities that post-training installed. Instead of treating behavior
preservation as a regularizer bolted onto next-token prediction, DiSC makes KL
divergence the primary objective: a frozen teacher reads a document prefix, the
trainable student does not, and training minimizes the divergence between their
distributions over the suffix. Both sides come from the same policy, so the
signal comes from richer conditioning rather than from a stronger model.

---

## Install

```bash
git clone <repo-url> && cd disc
pip install -r requirements.txt
pip install -e .
```

`vllm` and `lm-eval` are only needed for evaluation. `peft` is only needed for
the LoRA baseline, and `openai` only for rebuilding the BioASQ multiple-choice
set.

---

## Quickstart

Train DiSC on KUP and evaluate both halves of the objective:

```bash
# 1. adapt
python scripts/train.py --method disc --model qwen2.5-7b --corpus kup \
    --lr 3e-6 --softmax_temp 2.0 --num_splits 5 \
    --save_dir runs/qwen7b_kup_disc

# 2. knowledge adaptation (T_D)
python scripts/eval_domain.py --task kup \
    --model_path runs/qwen7b_kup_disc/step_5000 --data data/wikimcq_df.pickle

# 3. capability preservation (T_gen)
python scripts/eval_general.py --model_path runs/qwen7b_kup_disc/step_5000

# 4. table
python analysis/collect_results.py --runs_dir runs --task kup \
    --baseline runs/qwen7b_base --select_cp
```

---

## Methods

Every method is one flag on one script. They differ in objective, data, or
parameterization — never in file.

| Flag | Method | What changes |
|---|---|---|
| `--method ft` | Standard finetuning | Next-token prediction (Eq. 1) |
| `--method kl` | FT + KL regularization | CE + β·KL to the frozen initial policy (β = 0.1) |
| `--method lora` | FT + LoRA | r = 16, α = 32, all-linear, base weights frozen |
| `--method rephrase` | FT + Rephrase | Trains on the model's own paraphrases (on-policy) |
| `--method talr` | FT + TALR | Token-adaptive loss reweighting (Lin et al., 2025) |
| `--method cd_base` | CD-base | KL over generated transfer-set continuations (Padmanabhan et al., 2023) |
| `--method disc` | **DiSC** | KL over document suffixes with/without prefix conditioning (Eq. 3) |

Two baselines need a generation pass first:

```bash
python scripts/build_generations.py --mode rephrase --model qwen2.5-7b \
    --corpus kup --output data/rephrased_qwen7b_kup.jsonl

python scripts/build_generations.py --mode transfer --model qwen2.5-7b \
    --corpus kup --output data/transfer_set_qwen7b_kup.jsonl
```

DiSC needs neither; it takes suffixes from the document itself, which is what
makes it cheaper than CD-base.

### Split-strategy ablations

The choice of split points is a flag, so the ablation is a sweep rather than a
fork:

```bash
python scripts/train.py --method disc --split_strategy token_random \
    --model qwen2.5-3b --corpus kup --lr 3e-6 --save_dir runs/ablation_token_random
```

Available: `sentence_boundary` (default, the paper's method), `middle_only`,
`fixed_uniform_sentence`, `token_random`, `token_uniform`,
`token_random_variable_suffix`.

---

## Layout

```
src/disc/
  data.py         corpora (KUP, BioASQ), rephrases, transfer sets, replay
  models.py       loading, aliases, LoRA wrapping, optimizer, checkpointing
  splits.py       the six DiSC split strategies
  objectives.py   one function per method's loss
  trainer.py      the shared training loop (token-level and split-level)

scripts/
  train.py              all seven methods
  eval_domain.py        KUP + BioASQ multiple choice (vLLM, majority vote over n=5)
  eval_general.py       BBH, GPQA, MMLU-Pro, MuSR, IFEval, MATH, HumanEval via lm-eval
  compute_kl.py         per-token KL vs. the initial policy (Section 6.2)
  build_generations.py  rephrase / transfer-set generation
  build_bioasq_mcq.py   cloze → 4-way MCQ with GPT-5 distractors
  sweep.py              multi-GPU hyperparameter sweeps

analysis/
  collect_results.py    aggregation + capability-preserving checkpoint selection
```


---

## Reproducing the paper

For each model, use the LR that maximizes
adaptation on that corpus:

```bash
for m in ft kl talr lora rephrase; do
  python scripts/train.py --method $m --model qwen2.5-7b --corpus kup \
      --lr 1e-5 --save_dir runs/table2/qwen7b_kup_$m
done
```

Per-model LRs from Appendix B: KUP uses 1e-5 for Qwen-2.5-7B/3B, 4e-6 for
Llama-3.1-8B, 1.5e-5 for Qwen3-8B. BioASQ uses 1e-5 for Qwen-2.5-3B, 5e-6 for
Qwen-2.5-7B and Qwen3-8B, 1e-6 for Llama-3.1-8B.

## Citation

```bibtex
@article{padmanabhan2026disc,
  title  = {Updating Parametric Knowledge with Context Distillation
            Retains Post-Training Capabilities},
  author = {Padmanabhan, Shankar and Gul, Mustafa Omer and Goyal, Tanya},
  year   = {2026},
  eprint = {2602.16093},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
