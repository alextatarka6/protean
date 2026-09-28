"""
Format-agnostic tokenizer: word → integer index mapping.

Each battle format builds its own vocabulary (see protean/formats/<format>/vocab.py);
this module only holds the shared lookup/serialisation logic.

Usage:
    from protean.tokenizer import Tokenizer

    tok = Tokenizer.load("protean/data/gen1ou_vocab.json")
    ids = tok.tokenize("starmie surf water special")  # np.ndarray of int32
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

UNKNOWN_TOKEN: int = -1


def _clean(name: str) -> str:
    """Normalize a name to lowercase alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


class Tokenizer:
    """Simple word → integer vocabulary, same interface as metamon's PokemonTokenizer."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._vocab)

    def __getitem__(self, word: str) -> int:
        return self._vocab.get(word, UNKNOWN_TOKEN)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def all_words(self) -> list[str]:
        return list(self._vocab.keys())

    def _add(self, word: str) -> None:
        if word not in self._vocab:
            self._vocab[word] = len(self._vocab)

    def tokenize(self, text: str) -> np.ndarray:
        """Split on whitespace and map each word to its integer id."""
        return np.array([self._vocab.get(w, UNKNOWN_TOKEN) for w in text.split()], dtype=np.int32)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._vocab, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        tok = cls()
        with open(path) as f:
            tok._vocab = json.load(f)
        return tok
