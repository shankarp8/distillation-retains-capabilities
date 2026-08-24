"""Document splitting for DiSC (Step 1 of the algorithm).

Each strategy turns a document — supplied as a list of per-sentence token id
lists — into a list of ``(prefix_ids, suffix_ids)`` pairs. The default,
``sentence_boundary``, is the method described in the paper: sample ``k-1``
split points uniformly from the interior sentence boundaries and always
include the final sentence. The remaining strategies are the ablations used
to check that DiSC is not sensitive to the choice of split points.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

SplitPairs = List[Tuple[List[int], List[int]]]

MIN_SUFFIX_TOKENS = 2


def _flatten(sent_ids: List[List[int]]) -> List[int]:
    flat: List[int] = []
    for s in sent_ids:
        flat.extend(s)
    return flat


def _prefix_upto(sent_ids: List[List[int]], i: int) -> List[int]:
    ctx: List[int] = []
    for j in range(i):
        ctx.extend(sent_ids[j])
    return ctx


def _pairs_from_sentence_indices(sent_ids: List[List[int]], indices) -> SplitPairs:
    pairs: SplitPairs = []
    for i in indices:
        tgt = sent_ids[i]
        if len(tgt) < MIN_SUFFIX_TOKENS:
            continue
        pairs.append((_prefix_upto(sent_ids, i), tgt))
    return pairs


# --------------------------------------------------------------------------
# Strategies
# --------------------------------------------------------------------------


def sentence_boundary(sent_ids: List[List[int]], num_splits: int, rng: random.Random, **_) -> SplitPairs:
    """Paper default: k-1 random interior sentence boundaries plus the last sentence."""
    n = len(sent_ids)
    if n < 2:
        return []
    if num_splits == 1:
        indices = [n - 1]
    else:
        indices = rng.sample(range(1, n), min(num_splits - 1, n - 1))
        indices.append(n - 1)
        indices = sorted(set(indices))
    return _pairs_from_sentence_indices(sent_ids, indices)


def middle_only(sent_ids: List[List[int]], **_) -> SplitPairs:
    """Ablation: a single split at the middle sentence."""
    n = len(sent_ids)
    if n < 2:
        return []
    return _pairs_from_sentence_indices(sent_ids, [n // 2])


def fixed_uniform_sentence(sent_ids: List[List[int]], num_splits: int, **_) -> SplitPairs:
    """Ablation: evenly-spaced sentence-boundary splits."""
    n = len(sent_ids)
    if n < 2:
        return []
    k = min(num_splits, n - 1)
    step = (n - 1) / k
    indices = sorted({max(1, round(step * (i + 1))) for i in range(k)})
    return _pairs_from_sentence_indices(sent_ids, indices)


def token_random(sent_ids: List[List[int]], num_splits: int, rng: random.Random, **_) -> SplitPairs:
    """Ablation: random token-position splits, ignoring sentence boundaries.

    Suffixes are non-overlapping: each runs to the next split point or to the
    end of the document.
    """
    flat = _flatten(sent_ids)
    total = len(flat)
    if total < 4:
        return []
    k = min(num_splits, total - 2)
    positions = sorted(rng.sample(range(1, total - 1), k))
    return _pairs_from_positions(flat, positions)


def token_uniform(sent_ids: List[List[int]], num_splits: int, **_) -> SplitPairs:
    """Ablation: evenly-spaced token-position splits (deterministic)."""
    flat = _flatten(sent_ids)
    total = len(flat)
    if total < 4:
        return []
    k = min(num_splits, total - 2)
    step = total / (k + 1)
    positions = sorted({max(1, round(step * (i + 1))) for i in range(k)})
    return _pairs_from_positions(flat, positions)


def token_random_variable_suffix(
    sent_ids: List[List[int]],
    num_splits: int,
    rng: random.Random,
    suffix_tokens: int = 0,
    **_,
) -> SplitPairs:
    """Ablation: random token splits with a fixed suffix length.

    ``suffix_tokens=0`` lets every suffix run to the end of the document.
    """
    flat = _flatten(sent_ids)
    total = len(flat)
    if total < 4:
        return []
    k = min(num_splits, total - 2)
    positions = sorted(rng.sample(range(1, total - 1), k))

    pairs: SplitPairs = []
    for p in positions:
        end = min(p + suffix_tokens, total) if suffix_tokens > 0 else total
        tgt = flat[p:end]
        if len(tgt) < MIN_SUFFIX_TOKENS:
            continue
        pairs.append((flat[:p], tgt))
    return pairs


def _pairs_from_positions(flat: List[int], positions: List[int]) -> SplitPairs:
    pairs: SplitPairs = []
    total = len(flat)
    for idx, p in enumerate(positions):
        end = positions[idx + 1] if idx + 1 < len(positions) else total
        tgt = flat[p:end]
        if len(tgt) < MIN_SUFFIX_TOKENS:
            continue
        pairs.append((flat[:p], tgt))
    return pairs


STRATEGIES: Dict[str, Callable[..., SplitPairs]] = {
    "sentence_boundary": sentence_boundary,
    "middle_only": middle_only,
    "fixed_uniform_sentence": fixed_uniform_sentence,
    "token_random": token_random,
    "token_uniform": token_uniform,
    "token_random_variable_suffix": token_random_variable_suffix,
}


def make_splits(
    sent_ids: List[List[int]],
    strategy: str = "sentence_boundary",
    num_splits: int = 5,
    rng: random.Random | None = None,
    suffix_tokens: int = 0,
) -> SplitPairs:
    """Dispatch to a split strategy by name."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown split strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](
        sent_ids,
        num_splits=num_splits,
        rng=rng or random.Random(),
        suffix_tokens=suffix_tokens,
    )


def sentence_token_ids(document: str, tokenizer) -> List[List[int]]:
    """Split a document into sentences and tokenize each one."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    sentences = nltk.sent_tokenize(document)
    return [tokenizer(s, add_special_tokens=False)["input_ids"] for s in sentences]
