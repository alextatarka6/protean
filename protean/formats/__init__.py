"""
Format registry.

    from protean.formats import get_format
    fmt = get_format("gen1ou")

Format modules are imported lazily so offline code (e.g. BC training on the
gen1ou dataset) doesn't pay for formats it doesn't use.
"""
from __future__ import annotations

import importlib
from functools import lru_cache

from protean.formats.base import BattleFormat

# Showdown format id → "module:ClassName"
_REGISTRY: dict[str, str] = {
    "gen1ou": "protean.formats.gen1ou.format:Gen1OUFormat",
}

FORMAT_NAMES: list[str] = list(_REGISTRY)


@lru_cache(maxsize=None)
def get_format(name: str) -> BattleFormat:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown format {name!r}. Available: {FORMAT_NAMES}")
    module_name, cls_name = _REGISTRY[name].split(":")
    return getattr(importlib.import_module(module_name), cls_name)()


__all__ = ["BattleFormat", "FORMAT_NAMES", "get_format"]
