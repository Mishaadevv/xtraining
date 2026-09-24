"""A real byte-level BPE tokenizer, in pure Python.

This is used by the tiny backend (which has no external tokenizer), and it is
useful on its own: the UI can train a vocabulary on the user's own corpus and
then report exact token counts instead of character-based estimates.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .errors import ZxError
from .util import now_iso, read_json, write_json_atomic

SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
WORD_RE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
SPECIAL_RE = re.compile("|".join(re.escape(token) for token in SPECIAL_TOKENS))


def _count_words(texts: Iterable[str], max_words: int = 400_000) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for text in texts:
        for match in WORD_RE.finditer(text):
            counts[tuple(match.group().encode("utf-8"))] += 1
            if len(counts) >= max_words:
                return counts
    return counts


def _merge_pair(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    merged: list[int] = []
    index = 0
    while index < len(word):
        if index < len(word) - 1 and word[index] == pair[0] and word[index + 1] == pair[1]:
            merged.append(new_id)
            index += 2
        else:
            merged.append(word[index])
            index += 1
    return tuple(merged)


def train_bpe(
    texts: Iterable[str],
    vocab_size: int = 512,
    min_frequency: int = 2,
    progress: Any = None,
) -> dict[str, Any]:
    """Learn byte-pair merges from a corpus. Byte symbols occupy 0..255."""
    texts = list(texts)
    if not texts:
        raise ZxError(
            code="dataset_empty",
            message="Cannot train a tokenizer on an empty dataset.",
            hint="Import a dataset with real text before training a tokenizer.",
        )
    target_merges = max(0, int(vocab_size) - 256)
    counts = _count_words(texts)
    if not counts:
        raise ZxError(
            code="dataset_empty",
            message="The dataset contained no tokenisable characters.",
            hint="Check that the dataset is text and not only numeric columns.",
        )

    corpus: list[tuple[tuple[int, ...], int]] = list(counts.items())
    merges: list[list[int]] = []
    for step in range(target_merges):
        pair_counts: Counter[tuple[int, int]] = Counter()
        for word, frequency in corpus:
            for index in range(len(word) - 1):
                pair_counts[(word[index], word[index + 1])] += frequency
        if not pair_counts:
            break
        (best, frequency) = pair_counts.most_common(1)[0]
        if frequency < min_frequency:
            break
        new_id = 256 + len(merges)
        merges.append([int(best[0]), int(best[1])])
        corpus = [(_merge_pair(word, best, new_id), count) for word, count in corpus]
        if progress and step % 25 == 0:
            progress(step + 1, target_merges, frequency)
    return {
        "merges": merges,
        "special_tokens": list(SPECIAL_TOKENS),
        "vocab_size": 256 + len(merges) + len(SPECIAL_TOKENS),
        "trained_on": {
            "texts": len(texts),
            "unique_words": len(counts),
            "characters": sum(len(text) for text in texts),
        },
        "created_at": now_iso(),
    }


class BPETokenizer:
    """Encode/decode with a learned byte-level BPE vocabulary."""

    def __init__(self, payload: dict[str, Any]):
        self.merges: list[tuple[int, int]] = [tuple(pair) for pair in payload.get("merges", [])]
        self.specials: list[str] = list(payload.get("special_tokens", SPECIAL_TOKENS))
        self.vocab_size = int(payload.get("vocab_size") or (256 + len(self.merges) + len(self.specials)))
        self._ranks = {pair: index for index, pair in enumerate(self.merges)}
        self._expansion: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
        for index, pair in enumerate(self.merges):
            self._expansion[256 + index] = self._expansion[pair[0]] + self._expansion[pair[1]]
        self._special_ids = {
            token: 256 + len(self.merges) + index for index, token in enumerate(self.specials)
        }
        self._char_cache: dict[tuple[int, ...], list[int]] = {}

    # -- core -------------------------------------------------------------- #
    def _encode_word(self, word: bytes) -> list[int]:
        key = tuple(word)
        cached = self._char_cache.get(key)
        if cached is not None:
            return list(cached)
        symbols = list(key)
        while len(symbols) > 1:
            best_rank = None
            best_index = -1
            for index in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = index
            if best_rank is None:
                break
            pair = (symbols[best_index], symbols[best_index + 1])
            new_id = 256 + best_rank
            symbols = symbols[:best_index] + [new_id] + symbols[best_index + 2:]
        if len(self._char_cache) < 200_000:
            self._char_cache[key] = list(symbols)
        return symbols

    def encode(self, text: str, add_specials: bool = False) -> list[int]:
        ids: list[int] = []
        if add_specials:
            ids.append(self._special_ids.get("<|end|>", 0))
        segments = SPECIAL_RE.split(text)
        parts = SPECIAL_RE.findall(text)
        for index, segment in enumerate(segments):
            for match in WORD_RE.finditer(segment):
                ids.extend(self._encode_word(match.group().encode("utf-8")))
            if index < len(parts):
                ids.append(self._special_ids.get(parts[index], 0))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chunks: list[bytes] = []
        for token in ids:
            token = int(token)
            if token >= 256 + len(self.merges):
                special_index = token - (256 + len(self.merges))
                if 0 <= special_index < len(self.specials):
                    chunks.append(self.specials[special_index].encode("utf-8"))
                continue
            chunks.append(self._expansion.get(token, b""))
        return b"".join(chunks).decode("utf-8", errors="replace")

    def count(self, text: str) -> int:
        return len(self.encode(text))

    # -- persistence ------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "merges": [list(pair) for pair in self.merges],
            "special_tokens": self.specials,
            "vocab_size": self.vocab_size,
        }

    def save(self, path: Path) -> Path:
        write_json_atomic(Path(path), self.to_dict())
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        payload = read_json(Path(path), None)
        if not payload:
            raise ZxError(
                code="tokenizer_missing",
                message=f"Tokenizer file is missing or unreadable: {path}",
                hint="Train a tokenizer for this dataset, or pick a model that ships one.",
            )
        return cls(payload)


def stats_for_texts(tokenizer: BPETokenizer, texts: list[str]) -> dict[str, Any]:
    counts = [tokenizer.count(text) for text in texts]
    if not counts:
        return {"count": 0}
    ordered = sorted(counts)
    return {
        "count": len(counts),
        "total_tokens": sum(counts),
        "average": round(sum(counts) / len(counts), 1),
        "max": ordered[-1],
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[int(len(ordered) * 0.9)],
        "p99": ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
        "chars_per_token": round(sum(len(text) for text in texts) / max(1, sum(counts)), 2),
    }
