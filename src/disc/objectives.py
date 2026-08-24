"""Training objectives.

One function per method from the paper. All of them return a scalar loss.

  * :func:`ce_loss`          - standard finetuning (Eq. 1), also the base for
                               ``+LoRA`` and ``+Rephrase``, which change the
                               parameterization and the data respectively
                               rather than the objective.
  * :func:`kl_regularized_loss` - ``+KL``: CE plus beta * KL to the frozen
                               initial policy.
  * :func:`talr_loss`        - ``+TALR`` (Lin et al., 2025), token-adaptive
                               loss reweighting.
  * :func:`disc_loss`        - DiSC (Eq. 3): KL between the teacher conditioned
                               on the document prefix and the student
                               conditioned on nothing.
  * :func:`cd_base_loss`     - CD-base (Padmanabhan et al., 2023): same KL, but
                               the suffix is a generated transfer-set
                               continuation rather than a document sentence.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# Standard finetuning
# --------------------------------------------------------------------------


def ce_loss(model, batch) -> torch.Tensor:
    """Next-token prediction loss (Eq. 1)."""
    return model(**batch).loss


# --------------------------------------------------------------------------
# +KL regularization
# --------------------------------------------------------------------------


def kl_regularized_loss(
    model,
    teacher,
    batch,
    beta: float = 0.1,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """CE + ``beta`` * KL(teacher || student) on the training documents.

    ``beta=0.1`` is a deliberately strong regularization value relative to
    common practice, chosen to give the baseline the best chance of mitigating
    forgetting.
    """
    output = model(**batch)
    ce = output.loss
    student_logits = output.logits

    teacher_batch = {k: v.to(teacher.device) for k, v in batch.items()}
    with torch.no_grad():
        teacher_logits = teacher(**teacher_batch).logits

    mask = batch["attention_mask"].bool()
    student_masked = student_logits[:, :-1, :][mask[:, 1:]]
    teacher_masked = teacher_logits[:, :-1, :][mask[:, 1:]].to(student_logits.device)

    kl = F.kl_div(
        F.log_softmax(student_masked / temperature, dim=-1),
        F.softmax(teacher_masked / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)

    total = ce + beta * kl
    return total, {"ce_loss": ce.item(), "kl_loss": kl.item()}


# --------------------------------------------------------------------------
# +TALR (token-adaptive loss reweighting)
# --------------------------------------------------------------------------


def compute_dynamic_tau(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Median per-sequence NLL, used as the TALR temperature when tau is unset."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels != IGNORE_INDEX

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clone()
    safe_labels[~valid] = 0

    tgt_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    nll = -(tgt_log_probs * valid)
    per_seq_avg = nll.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    return max(per_seq_avg.median().detach().item(), 1e-6)


def talr_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tau: Optional[float] = None,
    min_prob: float = 1e-6,
    weight_floor: float = 0.01,
    normalize: str = "mean",
) -> torch.Tensor:
    """Token-adaptive loss reweighting (Lin et al., 2025).

    Down-weights high-loss tokens via ``w_t = clamp(p_t ** (1/tau), floor)``,
    with the floor preventing weight from collapsing onto a small subset of
    tokens. ``tau=None`` recomputes the dynamic tau each step.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels != IGNORE_INDEX
    if valid.sum() == 0:
        return shift_logits.sum() * 0

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.clone()
    safe_labels[~valid] = 0

    tgt_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    tgt_probs = torch.clamp(tgt_log_probs.exp(), min=min_prob)

    if tau is None:
        tau = compute_dynamic_tau(logits, labels)

    with torch.no_grad():
        weights = torch.clamp(tgt_probs.pow(1.0 / tau), min=weight_floor) * valid

    weighted_nll = weights * (-tgt_log_probs * valid)

    if normalize == "mean":
        return weighted_nll.sum() / valid.sum()
    if normalize == "sum":
        return weighted_nll.sum() / shift_logits.size(0)
    raise ValueError(f"Unknown normalize={normalize!r}")


# --------------------------------------------------------------------------
# Context distillation (DiSC and CD-base)
# --------------------------------------------------------------------------


@torch.no_grad()
def _teacher_suffix_logits(
    teacher,
    ctx_ids: List[int],
    tgt_ids: List[int],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Teacher sees prefix + suffix; returns logits over the suffix positions."""
    input_ids = torch.tensor([ctx_ids + tgt_ids], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        out = teacher(input_ids=input_ids, use_cache=False)
    logits = out.logits[0]
    ctx_len, tgt_len = len(ctx_ids), len(tgt_ids)
    return logits[ctx_len: ctx_len + tgt_len - 1, :]


def _student_suffix_logits(
    student,
    tgt_ids: List[int],
    device: torch.device,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Student sees the suffix only (no context conditioning)."""
    input_ids = torch.tensor([tgt_ids], dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=amp_dtype):
        out = student(input_ids=input_ids, use_cache=False)
    return out.logits[0][: len(tgt_ids) - 1, :]


def disc_loss(
    teacher,
    student,
    ctx_ids: List[int],
    tgt_ids: List[int],
    device: torch.device,
    temperature: float = 2.0,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """KL(teacher(suffix | prefix) || student(suffix)) for one split point.

    This is the per-split term inside Eq. 3; the caller averages over the
    ``|I|`` split points of a document.
    """
    if len(tgt_ids) < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    t_logits = _teacher_suffix_logits(teacher, ctx_ids, tgt_ids, device, amp_dtype)
    s_logits = _student_suffix_logits(student, tgt_ids, device, amp_dtype)

    t_probs = F.softmax(t_logits / temperature, dim=-1)
    s_logprobs = F.log_softmax(s_logits / temperature, dim=-1)
    return F.kl_div(s_logprobs, t_probs, reduction="batchmean") * (temperature ** 2)


def cd_base_loss(
    teacher,
    student,
    tokenizer,
    evidence: str,
    generation: str,
    device: torch.device,
    temperature: float = 1.0,
    max_generation_tokens: int = 1000,
    chunk_size: int = 64,
) -> Optional[torch.Tensor]:
    """CD-base: distill over a generated transfer-set continuation.

    Unlike DiSC, the suffix here is not part of the document — it is a
    continuation sampled from the teacher, which is what makes this baseline
    require an explicit (and expensive) generation pass. The KL is accumulated
    in chunks over the sequence to bound peak memory.
    """
    ev = tokenizer(evidence, return_tensors="pt", add_special_tokens=False).to(device)
    ge = tokenizer(generation, return_tensors="pt").to(device)

    if ge["input_ids"].shape[1] > max_generation_tokens:
        ge["input_ids"] = ge["input_ids"][:, :max_generation_tokens]
        ge["attention_mask"] = ge["attention_mask"][:, :max_generation_tokens]

    gen_ids = ge["input_ids"]
    if gen_ids.shape[1] < 2:
        return None

    teacher_input = torch.cat([ev["input_ids"], gen_ids], dim=1)
    attn = torch.ones_like(teacher_input, dtype=torch.long, device=device)

    with torch.no_grad():
        t_out = teacher(input_ids=teacher_input, attention_mask=attn, use_cache=False)
    ev_len, gen_len = ev["input_ids"].shape[1], gen_ids.shape[1]
    t_logits = t_out.logits[:, ev_len: ev_len + gen_len - 1, :]

    s_attn = torch.ones_like(gen_ids, dtype=torch.long, device=device)
    s_logits = student(input_ids=gen_ids, attention_mask=s_attn, use_cache=False).logits[:, :-1, :]

    total, count = 0.0, 0
    for start in range(0, t_logits.shape[1], chunk_size):
        end = min(start + chunk_size, t_logits.shape[1])
        t_probs = F.softmax(t_logits[:, start:end, :] / temperature, dim=-1)
        s_logprobs = F.log_softmax(s_logits[:, start:end, :] / temperature, dim=-1)
        total = total + F.kl_div(s_logprobs, t_probs, reduction="batchmean") * (temperature ** 2)
        count += 1

    return total / max(count, 1)
