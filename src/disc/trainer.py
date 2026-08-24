"""The single training loop shared by every method.

Two loop shapes are needed:

  * **token-level** (``ft``, ``kl``, ``talr``, ``lora``, ``rephrase``, and
    replay variants) — iterate over tokenized documents, one optimizer step
    per document.
  * **split-level** (``disc``, ``cd_base``) — iterate over raw documents,
    accumulate gradients across the ``|I|`` split points of that document,
    then take one optimizer step.

Both share checkpointing, logging, and optimizer construction.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import objectives
from .data import EvidenceDataset, RawTextDataset, make_collate_fn
from .models import build_optimizer, save_checkpoint
from .splits import make_splits, sentence_token_ids

TOKEN_LEVEL_METHODS = {"ft", "kl", "talr", "lora", "rephrase"}
SPLIT_LEVEL_METHODS = {"disc", "cd_base"}
ALL_METHODS = sorted(TOKEN_LEVEL_METHODS | SPLIT_LEVEL_METHODS)


@dataclass
class TrainConfig:
    """Everything that varies between runs, in one place."""

    method: str = "ft"
    model_name: str = "qwen2.5-7b"
    corpus: str = "kup"
    save_dir: str = "runs/default"

    lr: float = 1e-5
    weight_decay: float = 0.01
    seed: int = 1234
    save_every: int = 0          # 0 = only save at the end
    max_steps: int = 0           # 0 = one full epoch
    max_length: Optional[int] = None

    # +KL
    kl_beta: float = 0.1
    kl_temperature: float = 1.0

    # +TALR
    talr_tau: Optional[float] = None   # None = dynamic tau per step
    talr_normalize: str = "mean"

    # +LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # +Rephrase
    rephrase_path: Optional[str] = None

    # DiSC
    softmax_temp: float = 2.0
    num_splits: int = 5
    split_strategy: str = "sentence_boundary"
    suffix_tokens: int = 0

    # CD-base
    transfer_path: Optional[str] = None

    # Replay (post-submission experiments)
    replay_source: Optional[str] = None
    replay_n: int = 0
    replay_kl_weight: float = 0.0

    device: str = "cuda:0"
    amp_dtype: str = "bfloat16"

    def torch_amp_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16}[self.amp_dtype]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_config(cfg: TrainConfig, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        model,
        tokenizer,
        documents: List[str],
        teacher=None,
        replay_flags: Optional[List[bool]] = None,
        transfer_map: Optional[dict] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.documents = documents
        self.teacher = teacher
        self.replay_flags = replay_flags
        self.transfer_map = transfer_map or {}

        self.device = torch.device(cfg.device)
        self.optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)
        self.global_step = 0
        self.losses: List[float] = []
        self.is_lora = cfg.method == "lora"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> str:
        write_config(self.cfg, self.cfg.save_dir)
        if self.cfg.method in SPLIT_LEVEL_METHODS:
            self._train_split_level()
        else:
            self._train_token_level()

        final = save_checkpoint(
            self.model, self.tokenizer, self.cfg.save_dir, self.global_step, self.is_lora
        )
        self._dump_losses()
        return final

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    def _train_token_level(self) -> None:
        dataset = EvidenceDataset(self.documents, self.tokenizer, self.cfg.max_length)
        loader = DataLoader(
            dataset,
            batch_size=1,
            pin_memory=True,
            num_workers=2,
            collate_fn=make_collate_fn(self.tokenizer),
        )
        total = self.cfg.max_steps or len(dataset)
        bar = tqdm(loader, total=total, desc=f"train[{self.cfg.method}]", dynamic_ncols=True)

        for i, batch in enumerate(bar):
            if self.cfg.max_steps and self.global_step >= self.cfg.max_steps:
                break

            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad(set_to_none=True)

            extra: dict = {}
            is_replay = bool(self.replay_flags[i]) if self.replay_flags else False

            if self.cfg.method == "talr":
                logits = self.model(**batch).logits
                loss = objectives.talr_loss(
                    logits,
                    batch["labels"],
                    tau=self.cfg.talr_tau,
                    normalize=self.cfg.talr_normalize,
                )
            elif self.cfg.method == "kl":
                loss, extra = objectives.kl_regularized_loss(
                    self.model, self.teacher, batch,
                    beta=self.cfg.kl_beta, temperature=self.cfg.kl_temperature,
                )
            elif is_replay and self.cfg.replay_kl_weight > 0:
                # Replay ablation: anchor the model to the initial policy on
                # replay demonstrations only, leaving adaptation data on CE.
                loss, extra = objectives.kl_regularized_loss(
                    self.model, self.teacher, batch,
                    beta=self.cfg.replay_kl_weight, temperature=self.cfg.kl_temperature,
                )
            else:
                loss = objectives.ce_loss(self.model, batch)

            loss.backward()
            self.optimizer.step()
            self._step_end(bar, loss, extra)

    def _train_split_level(self) -> None:
        dataset = RawTextDataset(self.documents)
        loader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=2)
        rng = random.Random(self.cfg.seed)
        amp_dtype = self.cfg.torch_amp_dtype()

        total = self.cfg.max_steps or len(dataset)
        bar = tqdm(loader, total=total, desc=f"train[{self.cfg.method}]", dynamic_ncols=True)

        for batch in bar:
            if self.cfg.max_steps and self.global_step >= self.cfg.max_steps:
                break

            doc_idx = int(batch["idx"][0])
            document = batch["text"][0]

            if self.cfg.method == "cd_base":
                loss = self._cd_base_step(doc_idx)
                if loss is None:
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                self._step_end(bar, loss)
                continue

            # DiSC: accumulate over the document's split points, one step per doc.
            sent_ids = sentence_token_ids(document, self.tokenizer)
            pairs = make_splits(
                sent_ids,
                strategy=self.cfg.split_strategy,
                num_splits=self.cfg.num_splits,
                rng=rng,
                suffix_tokens=self.cfg.suffix_tokens,
            )
            if not pairs:
                self.global_step += 1
                continue

            self.optimizer.zero_grad(set_to_none=True)
            last_loss = None
            for ctx_ids, tgt_ids in pairs:
                loss = objectives.disc_loss(
                    self.teacher, self.model, ctx_ids, tgt_ids,
                    device=self.device,
                    temperature=self.cfg.softmax_temp,
                    amp_dtype=amp_dtype,
                ) / len(pairs)
                loss.backward()
                last_loss = loss
                self.losses.append(float(loss.item()))

            self.optimizer.step()
            self._step_end(bar, last_loss, record_loss=False)

    def _cd_base_step(self, doc_idx: int):
        item = self.transfer_map.get(doc_idx)
        if not item:
            return None
        generations = item.get("generations") or []
        if not generations:
            return None
        return objectives.cd_base_loss(
            self.teacher, self.model, self.tokenizer,
            evidence=item["evidence"],
            generation=generations[0],
            device=self.device,
            temperature=self.cfg.softmax_temp,
        )

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    def _step_end(self, bar, loss, extra: Optional[dict] = None, record_loss: bool = True) -> None:
        self.global_step += 1
        if loss is not None and record_loss:
            self.losses.append(float(loss.item()))

        postfix = {"step": self.global_step}
        if loss is not None:
            postfix["loss"] = round(float(loss.item()), 4)
        if extra:
            postfix.update({k: round(v, 4) for k, v in extra.items()})
        bar.set_postfix(postfix)

        if self.cfg.save_every and self.global_step % self.cfg.save_every == 0:
            save_checkpoint(
                self.model, self.tokenizer, self.cfg.save_dir, self.global_step, self.is_lora
            )
            self._dump_losses()

    def _dump_losses(self) -> None:
        with open(os.path.join(self.cfg.save_dir, "losses.txt"), "w") as f:
            f.write("\n".join(map(str, self.losses)))
