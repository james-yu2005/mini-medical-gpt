# MedLM — Transformers from Scratch

A from-scratch language modeling project in PyTorch, inspired by [Karpathy's "Let's build GPT" series](https://www.youtube.com/watch?v=kCc8FmEb1nY). Trains on medical flashcard text with a progression from simple bigram models to a full GPT with a byte-level BPE tokenizer.

## What this project does

1. **Bigram model** — predicts the next character from only the previous one
2. **GPT + BPE** — a ~10M-parameter Transformer trained on subword tokens (byte-level BPE, GPT-2 style)
3. **Custom tokenizer** — BPE trained from scratch on UTF-8 bytes (no HuggingFace tokenizer)

## Project structure

```
transformers/
├── data/
│   └── input.txt          # training corpus (generated, not committed)
├── tokenizer/
│   ├── bpe_tokenizer.py   # byte-level BPE encode/decode/train
│   ├── corpus.py          # paths, corpus loading, token cache
│   ├── train.py           # train tokenizer + encode corpus
│   ├── bpe.json           # learned merges (generated)
│   └── corpus_tokens.pt   # pre-encoded training data (generated)
├── bigram.py              # step 1: character bigram baseline
├── gpt_bpe.py             # step 2: GPT on BPE tokens (main model)
└── test_encode.py         # interactive tokenizer tester
```

## Requirements

- Python 3.10+
- PyTorch
- `datasets` (only for downloading the medical corpus)

```powershell
pip install torch datasets
```

## Quick start

Run everything from the project root (`transformers/`).

### 1. Get training data

Place your text at `data/input.txt`, or download the medical flashcard dataset:

```powershell
python data/load_dataset.py
```

This writes ~50k medical explanations to `data/input.txt`.

### 2. Train tokenizer and encode corpus

```powershell
python -m tokenizer.train
```

This will:
- Train byte-level BPE on the first **2M characters** (768 merges → 1024 vocab)
- Encode the first **4M characters** for GPT training
- Save `tokenizer/bpe.json` and `tokenizer/corpus_tokens.pt`

To re-encode only (tokenizer already exists):

```powershell
python -m tokenizer.train --cache-only
```

### 3. Test the tokenizer (optional)

```powershell
python test_encode.py
```

### 4. Train models

```powershell
# Baseline: character bigram
python bigram.py

# Main model: GPT on BPE tokens (~10M params)
python gpt_bpe.py
```

`gpt_bpe.py` loads cached tokens automatically and generates text from the prompt `"Diabetes is"`.

## Model details

### Bigram (`bigram.py`)

| Setting | Value |
|---------|-------|
| Token level | Character |
| Context | 8 chars |
| Parameters | ~4K |

### GPT (`gpt_bpe.py`)

| Setting | Value |
|---------|-------|
| Token level | Byte BPE (1024 vocab) |
| Context (`block_size`) | 256 tokens |
| `n_embd` | 384 |
| Layers / heads | 6 / 6 |
| Parameters | ~10M |
| Training steps | 5000 |

### BPE tokenizer

| Setting | Value |
|---------|-------|
| Type | Byte-level BPE (raw UTF-8) |
| Base vocab | 256 bytes |
| Merges | 768 |
| Train sample | First 2M chars of corpus |
| LM corpus | First 4M chars of corpus |

Limits are in `tokenizer/corpus.py` (`TRAIN_SAMPLE_CHARS`, `CORPUS_MAX_CHARS`).

## How BPE fits in

```
text  →  UTF-8 bytes [0–255]  →  apply merges  →  token IDs  →  GPT
IDs   →  lookup bytes per ID  →  UTF-8 decode   →  text
```

- `bpe.json` — merge rules (required for encode/decode)
- `corpus_tokens.pt` — pre-encoded integer sequence for fast training startup

## Learning progression

```
bigram.py          gpt_bpe.py
    │                  │
    │                  ├── tokenizer/bpe_tokenizer.py
    │                  ├── self-attention (6 heads)
    │                  ├── 6 transformer blocks
    │                  └── byte BPE subwords
    │
    └── 1-char memory, embedding lookup only
```

## Generated files (gitignored)

These are created locally and not committed:

- `data/input.txt`
- `tokenizer/bpe.json`
- `tokenizer/corpus_tokens.pt`
- `gpt_venv/`

## Resume bullet

> Built a byte-level BPE tokenizer and ~10M-parameter GPT from scratch in PyTorch; trained on medical flashcard corpus with causal self-attention, checkpointed token cache, and subword-level generation.
