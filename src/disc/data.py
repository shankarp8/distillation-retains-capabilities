"""Adaptation corpora and dataset wrappers.

Consolidates the dataset-loading logic that was previously copy-pasted (and
commented in/out) across ~150 training scripts. Every corpus is exposed
through :func:`load_corpus`, which returns a plain ``List[str]`` of documents.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from torch.utils.data import Dataset

# --------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------

KUP_HF_NAME = "aochongoliverli/KUP"
BIOASQ_HF_NAME = "kroshan/BioASQ"


def load_kup() -> List[str]:
    """KUP (Li & Goyal, 2025): 5k synthetic news-style knowledge updates."""
    from datasets import load_dataset

    ds = load_dataset(KUP_HF_NAME)["train"]
    return [ex["evidence_news"] for ex in ds]


def load_bioasq() -> List[str]:
    """BioASQ (Krithara et al., 2023) documents.

    The HF release packs the document into a single ``text`` field behind a
    ``<context>`` marker; everything after the marker is the document body.
    """
    from datasets import load_dataset

    ds = load_dataset(BIOASQ_HF_NAME)["train"]
    docs = []
    for ex in ds:
        text = ex["text"]
        marker = "<context>"
        idx = text.find(marker)
        docs.append(text[idx + len(marker):] if idx != -1 else text)
    return docs


def load_jsonl(path: str | Path) -> List[Dict]:
    """Read a .jsonl file into a list of dicts, skipping blank lines."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(records: Sequence[Dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_rephrased(path: str | Path) -> List[str]:
    """On-policy rephrases produced by ``scripts/build_rephrases.py``.

    Used by the ``+Rephrase`` baseline (self-distillation, Yang et al. 2024b):
    the model is finetuned on its own paraphrases of each document rather than
    on the raw text.
    """
    return [rec["generations"][0] for rec in load_jsonl(path)]


def load_transfer_map(path: str | Path) -> Dict[int, Dict]:
    """Transfer set for the CD-base baseline (Padmanabhan et al., 2023).

    Maps document index -> {"evidence": str, "generations": List[str]}.
    """
    mapping: Dict[int, Dict] = {}
    for obj in load_jsonl(path):
        try:
            key = int(obj["id"])
        except (KeyError, TypeError, ValueError):
            key = obj.get("id")
        mapping[key] = obj
    return mapping


CORPUS_LOADERS = {
    "kup": load_kup,
    "bioasq": load_bioasq,
}


def load_corpus(
    name: str,
    rephrase_path: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 1234,
) -> List[str]:
    """Return the adaptation documents for ``name``.

    If ``rephrase_path`` is given, the on-policy rephrases replace the raw
    documents (this is what distinguishes ``+Rephrase`` from plain FT).
    """
    if rephrase_path:
        docs = load_rephrased(rephrase_path)
    elif name in CORPUS_LOADERS:
        docs = CORPUS_LOADERS[name]()
    else:
        raise ValueError(
            f"Unknown corpus {name!r}; expected one of {sorted(CORPUS_LOADERS)} "
            "or pass --rephrase_path for a rephrased corpus."
        )

    if shuffle:
        random.Random(seed).shuffle(docs)
    return docs


# --------------------------------------------------------------------------
# Torch datasets
# --------------------------------------------------------------------------


class EvidenceDataset(Dataset):
    """Tokenized documents for next-token-prediction training (FT/KL/TALR/LoRA)."""

    def __init__(self, texts: Sequence[str], tokenizer, max_length: Optional[int] = None):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        kwargs = {"return_tensors": "pt"}
        if self.max_length is not None:
            kwargs.update(truncation=True, max_length=self.max_length)
        enc = self.tokenizer(self.texts[idx], **kwargs)

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class RawTextDataset(Dataset):
    """Untokenized documents, for methods that tokenize per split (DiSC/CD-base)."""

    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return {"text": self.texts[idx], "idx": idx}


def make_collate_fn(tokenizer):
    """Pad a batch of tokenized examples."""

    def collate(batch):
        return tokenizer.pad(batch, padding=True, return_tensors="pt")

    return collate


# --------------------------------------------------------------------------
# Replay corpora (post-submission experiments)
# --------------------------------------------------------------------------


def load_replay(source: str, n: int, seed: int = 1234) -> List[str]:
    """Replay demonstrations mixed into the adaptation stream.

    ``source`` is one of ``gsm8k``, ``math``, ``alpaca``, or a path to a
    .jsonl file with ``prompt``/``response`` fields.
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    if source == "gsm8k":
        ds = load_dataset("gsm8k", "main")["train"]
        pairs = [(e["question"], e["answer"]) for e in ds]
    elif source == "math":
        pairs = []
        for subject in ("algebra", "counting_and_probability", "precalculus", "number_theory"):
            ds = load_dataset("EleutherAI/hendrycks_math", subject)["train"]
            pairs.extend((e["problem"], e["solution"]) for e in ds)
    elif source == "alpaca":
        ds = load_dataset("yahma/alpaca-cleaned")["train"]
        pairs = [
            (
                f"### Instruction:\n{e['instruction']}\n\n"
                + (f"### Input:\n{e['input']}\n\n" if e.get("input") else "")
                + "### Response:\n",
                e["output"],
            )
            for e in ds
        ]
    else:
        records = load_jsonl(source)
        pairs = [(r["prompt"], r["response"]) for r in records]

    if n < len(pairs):
        pairs = rng.sample(pairs, n)
    return [f"{q}\n{a}" for q, a in pairs]
