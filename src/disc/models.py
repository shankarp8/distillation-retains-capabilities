"""Model loading, LoRA wrapping, and checkpointing."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Short aliases so experiment commands stay readable. Override with a real
# path or HF id at the command line; unknown names are passed through as-is.
MODEL_ALIASES = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3-8b": "Qwen/Qwen3-8B",
}


def resolve_model(name: str) -> str:
    """Map an alias to a model path, or return ``name`` unchanged."""
    return MODEL_ALIASES.get(name, name)


def load_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(resolve_model(name))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    name: str,
    device: torch.device | str = "cuda",
    dtype: Optional[torch.dtype] = None,
    gradient_checkpointing: bool = False,
):
    """Load a causal LM for training.

    Defaults to fp32 (as in the paper's experiments); pass ``dtype`` to
    override, e.g. ``torch.bfloat16`` for frozen teachers.
    """
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(resolve_model(name), **kwargs)
    model.to(device)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def load_frozen_teacher(
    name: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load a frozen copy of the initial post-trained model.

    Used as the reference policy for ``+KL`` and as the context-conditioned
    teacher for DiSC and CD-base. The teacher is never updated.
    """
    teacher = load_model(name, device=device, dtype=dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def apply_lora(
    model,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: str = "all-linear",
):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def build_optimizer(model, lr: float, weight_decay: float = 0.01):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )


def save_checkpoint(model, tokenizer, save_dir: str, step: int, is_lora: bool = False) -> str:
    """Save a checkpoint at ``{save_dir}/step_{step}``.

    For LoRA runs the adapter is saved separately under ``lora_modules/`` and a
    merged full model is written alongside it, so downstream evaluation can
    load either one with ``from_pretrained``.
    """
    path = os.path.join(save_dir, f"step_{step}")
    os.makedirs(path, exist_ok=True)

    if is_lora:
        model.save_pretrained(os.path.join(path, "lora_modules"))
        merged = model.merge_and_unload()
        merged.save_pretrained(path)
    else:
        model.save_pretrained(path)

    tokenizer.save_pretrained(path)
    print(f"[step {step}] saved checkpoint -> {path}")
    return path


def load_model_and_tokenizer(name: str, **kwargs) -> Tuple:
    return load_model(name, **kwargs), load_tokenizer(name)
