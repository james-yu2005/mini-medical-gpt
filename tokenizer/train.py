"""Train byte-level BPE tokenizer and cache corpus tokens."""

import argparse

from .bpe_tokenizer import BPETokenizer
from .corpus import (
    BPE_VOCAB_SIZE,
    CORPUS_MAX_CHARS,
    CORPUS_PATH,
    TOKENIZER_PATH,
    TRAIN_SAMPLE_CHARS,
    read_corpus,
    save_corpus_cache,
)


def train_tokenizer() -> BPETokenizer:
    text = read_corpus()
    corpus_text = text[:CORPUS_MAX_CHARS]
    print(f"Corpus: {len(text):,} characters (LM uses first {len(corpus_text):,})")
    print(f"Training byte-level BPE on first {TRAIN_SAMPLE_CHARS:,} characters...")

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=BPE_VOCAB_SIZE, max_chars=TRAIN_SAMPLE_CHARS)
    tokenizer.save(TOKENIZER_PATH)

    sample = text[:200]
    encoded = tokenizer.encode(sample)
    print(f"Vocabulary size: {tokenizer.vocab_size}  (256 bytes + {len(tokenizer.merges)} merges)")
    print(f"Saved to:        {TOKENIZER_PATH}")

    return tokenizer


def cache_corpus(tokenizer: BPETokenizer | None = None) -> None:
    tokenizer = tokenizer or BPETokenizer.load(TOKENIZER_PATH)
    corpus_text = read_corpus(CORPUS_MAX_CHARS)
    print("Encoding corpus subset...", flush=True)
    tokens = tokenizer.encode(corpus_text)
    save_corpus_cache(tokens, CORPUS_MAX_CHARS)
    print(f"Saved {len(tokens):,} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer and cache corpus tokens.")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Re-encode corpus using existing bpe.json (skip merge training).",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}. Run: python data/load_dataset.py")

    if args.cache_only:
        cache_corpus()
    else:
        tokenizer = train_tokenizer()
        print()
        cache_corpus(tokenizer)


if __name__ == "__main__":
    main()
