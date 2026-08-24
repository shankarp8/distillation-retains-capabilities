# Data

Generated artifacts live here and are gitignored. Rebuild them with:

| File | Built by | Used by |
|---|---|---|
| `rephrased_{model}_{corpus}.jsonl` | `scripts/build_generations.py --mode rephrase` | `train.py --method rephrase` |
| `transfer_set_{model}_{corpus}.jsonl` | `scripts/build_generations.py --mode transfer` | `train.py --method cd_base` |
| `bioasq_mcq.jsonl` | `scripts/build_bioasq_mcq.py` | `eval_domain.py --task bioasq` |
| `wikimcq_df.pickle` | released with KUP (Li & Goyal, 2025) | `eval_domain.py --task kup` |

The adaptation corpora themselves (KUP, BioASQ) are pulled from the Hugging
Face Hub at load time and are not stored here.
