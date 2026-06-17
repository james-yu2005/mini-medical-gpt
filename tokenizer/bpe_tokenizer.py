"""
Byte Pair Encoding (BPE) on raw UTF-8 bytes — GPT-2 style.

Algorithm:
  1. Text → UTF-8 bytes (token IDs 0-255)
  2. Repeatedly merge the most frequent adjacent pair
  3. New merged tokens get IDs 256, 257, ...
  4. Encode applies the same merges in training order
"""

import json
from collections import defaultdict
from pathlib import Path

BYTE_BASE = 256
TOKENIZER_VERSION = 2


def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    a, b = pair
    i = 0
    out: list[int] = []
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for i in range(len(ids) - 1):
        counts[(ids[i], ids[i + 1])] += 1
    return counts


def _pair_exists(ids: list[int], pair: tuple[int, int]) -> bool:
    a, b = pair
    for i in range(len(ids) - 1):
        if ids[i] == a and ids[i + 1] == b:
            return True
    return False


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = {}
        self._rebuild_vocab()

    @property
    def vocab_size(self) -> int:
        return BYTE_BASE + len(self.merges)

    @property
    def itos(self) -> dict[int, str]:
        return {i: self.token_to_str(i) for i in range(self.vocab_size)}

    def token_to_str(self, token_id: int) -> str:
        raw = self.vocab[token_id]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"<0x{raw.hex()}>"

    def _rebuild_vocab(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(BYTE_BASE)}
        for idx, (a, b) in enumerate(self.merges):
            new_id = BYTE_BASE + idx
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    def train(self, text: str, vocab_size: int, max_chars: int | None = None) -> None:
        if max_chars is not None:
            text = text[:max_chars]

        ids = list(text.encode("utf-8"))
        num_merges = max(0, vocab_size - BYTE_BASE)
        self.merges = []

        print(f"  training on {len(ids):,} bytes...", flush=True)

        for step in range(num_merges):
            counts = _count_pairs(ids)
            if not counts:
                break

            best_pair = max(counts, key=counts.get)
            if counts[best_pair] < 2:
                break

            new_id = BYTE_BASE + len(self.merges)
            self.merges.append(best_pair)
            ids = _merge_ids(ids, best_pair, new_id)

            if (step + 1) % 100 == 0 or step + 1 == num_merges:
                print(
                    f"  merge {step + 1}/{num_merges}  pair={best_pair}  "
                    f"bytes=({best_pair[0]:#04x}, {best_pair[1]:#04x})  freq={counts[best_pair]}",
                    flush=True,
                )

        self._rebuild_vocab()

    def _apply_merges(self, ids: list[int]) -> list[int]:
        for idx, pair in enumerate(self.merges):
            if not _pair_exists(ids, pair):
                continue
            ids = _merge_ids(ids, pair, BYTE_BASE + idx)
        return ids

    def encode(self, text: str) -> list[int]:
        return self._apply_merges(list(text.encode("utf-8")))

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": TOKENIZER_VERSION,
            "merges": [list(pair) for pair in self.merges],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != TOKENIZER_VERSION:
            raise ValueError(
                "Outdated tokenizer format. Re-train with: python -m tokenizer.train"
            )
        tok = cls()
        tok.merges = [tuple(pair) for pair in payload["merges"]]
        tok._rebuild_vocab()
        return tok
