"""DiSC: Distillation via Split Contexts.

Continual knowledge adaptation of post-trained LMs that learns new knowledge
from a document corpus while retaining instruction-following, reasoning, and
coding capabilities acquired during post-training.

Submodules are imported lazily so that light-weight pieces (e.g. ``splits``)
can be used without pulling in torch, transformers, and vLLM.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "1.0.0"

_SUBMODULES = ("data", "models", "objectives", "splits", "trainer")

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from . import data, models, objectives, splits, trainer  # noqa: F401


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_SUBMODULES))


__all__ = list(_SUBMODULES)
