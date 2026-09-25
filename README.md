# Character-Level GPT from Scratch

A PyTorch implementation of a decoder-only, character-level GPT language model, built from the ground up (custom self-attention, multi-head attention, and transformer blocks — no `nn.Transformer` or Hugging Face). Includes a small proof-of-concept trained on *Dracula* and a larger version trained on OpenWebText, plus an interactive chat script for sampling from the trained model.

## Project Structure

| File | Description |
|---|---|
| `llm.ipynb` | Starter notebook. Trains a simple bigram language model on `dracula.txt` to validate the training loop and data pipeline before scaling up. |
| `gpt-v1.ipynb` | Full GPT model (self-attention, multi-head attention, transformer blocks) trained on the OpenWebText corpus. Notebook version of the main model. |
| `training.py` | Original script version of the GPT training loop. Loads `model-01.pkl` and continues training with a flat learning rate. Kept for reference; `train_v2.py` supersedes it. |
| `train_v2.py` | Improved training script: trains a fresh model on the cleaned vocab, `block_size=256`, warmup + cosine LR decay, gradient clipping, time-boxed with periodic checkpointing to `model-0N.pkl` / `model-0N-best.pkl`. |
| `chatbox.py` | Interactive command-line chat interface. Loads the current best checkpoint and generates text completions from user prompts. |
| `evaluate.py` / `evaluate_v2.py` | Non-interactive evaluation: parameter counts, validation loss (nats + bits/char) averaged over many random batches, and sample generations at multiple temperatures. |
| `build_vocab.py` | Builds a cleaned character vocabulary by sampling the corpus and keeping only characters above a frequency threshold (see *Known issues found & fixed* below). |
| `data-extract.py` | Original OpenWebText extraction script. **Has a bug** — see below. Kept for reference. |
| `data-extract-v2.py` | Corrected extraction script that properly parses the `.xz` archives as tar files instead of raw text. |
| `torch-examples.ipynb` | Scratch notebook of PyTorch fundamentals (tensor ops, embeddings, softmax, matrix multiplication, CPU vs. GPU benchmarking) used while learning the building blocks for the model. |
| `dracula.txt` | Small text corpus (public-domain novel) used for the early bigram prototype. |
| `model-01.pkl` | Original checkpoint (53M params, 32k-char vocab). Not tracked in git (see `.gitignore`) — too large for GitHub. |
| `model-02-best.pkl` | Best checkpoint from `train_v2.py` (14.5M params, cleaned 254-char vocab, `block_size=256`). Not tracked in git. |

## Evaluation & fixes (this pass)

The original `model-01.pkl` was evaluated end-to-end (parameter breakdown, 200-batch validation loss, sample generations) and three issues were found and fixed:

1. **Bloated vocabulary.** `vocab.txt` was built by dumping every unique character seen across the entire 46GB corpus — 32,172 characters, mostly one-off Unicode noise. This meant **46.6% of the model's 53M parameters** were spent on the token embedding + output head alone. `build_vocab.py` now samples the corpus and keeps only characters covering 99.98% of observed frequency mass (254 characters), which cut total parameters to 14.5M while *improving* validation loss.
2. **Short context.** `block_size` was 128 characters (~20-25 words) — too short for topical coherence. Increased to 256 in `train_v2.py`.
3. **No LR schedule / short training.** The original loop used a flat `lr=3e-4` for a fixed 5000 iterations. `train_v2.py` adds linear warmup + cosine decay (decay driven by wall-clock progress through a time budget, so it's robust to unknown steps/sec) and gradient clipping, and trains until a time budget is reached (checkpointing periodically so no progress is lost).

A fourth, more serious issue was found while testing the retrained model:

4. **Corpus corruption from a tar-parsing bug.** The original OpenWebText `.xz` archives are actually **compressed tar files** (each contains hundreds of individual `<hash>.txt` documents). `data-extract.py` decoded the raw decompressed byte stream directly as UTF-8 text instead of parsing it as a tar archive, so `train_split.txt` / `val_split.txt` were contaminated throughout with tar headers and NUL padding (~11% of all bytes in a sampled region were literal `\x00`, plus tar entry names like `0999049-cf978a7f....txt` leaking in as if they were prose). `data-extract-v2.py` fixes this by opening each archive with Python's `tarfile` module and extracting only the genuine per-document text, separated by an explicit document-boundary character. This is being used to rebuild the corpus and retrain again.

| | model-01 (original) | model-02-best (vocab+context+schedule fix) |
|---|---|---|
| Parameters | 53,163,180 | 14,480,894 |
| `vocab_size` | 32,172 | 254 |
| `block_size` | 128 | 256 |
| Val loss | 1.3095 nats | 1.0182 nats |
| Bits/char | 1.889 | 1.469 |

A further retrain on the corrected (tar-parsed) corpus is in progress; results will supersede the table above once complete.

## Model Architecture

`GPTLLM` is a decoder-only transformer with:

- **Token + positional embeddings** — learned embeddings for each character and each position in the sequence.
- **Stacked transformer blocks** (`n_layer = 8`), each containing:
  - Multi-head causal self-attention (`n_head = 8`), with a lower-triangular mask so each token only attends to previous tokens.
  - A position-wise feed-forward network (4x expansion, ReLU, projection back down).
  - Residual connections and layer normalization around both sub-layers.
- **Final layer norm** + a linear head projecting to vocabulary logits.
- **Dropout** (`0.2`) applied in attention and feed-forward layers for regularization.

### Hyperparameters

| Parameter | `training.py` (original) | `train_v2.py` (current) |
|---|---|---|
| `batch_size` | 64 | 64 |
| `block_size` (context length) | 128 | 256 |
| `n_embd` (embedding dim) | 384 | 384 |
| `n_layer` | 8 | 8 |
| `n_head` | 8 | 8 |
| `dropout` | 0.2 | 0.2 |
| `learning_rate` | flat 3e-4 | warmup to 3e-4, cosine decay to 3e-5 |
| stopping condition | fixed 5000 iters | time-boxed (default 4h), checkpointed periodically |

Tokenization is **character-level**: `build_vocab.py` derives a frequency-cleaned vocabulary from the corpus, with simple `encode`/`decode` lookup dictionaries (no BPE/subword tokenization) plus a reserved UNK character for anything outside the vocab.

## Requirements

- Python 3.9+
- PyTorch (with CUDA support recommended — training falls back to CPU automatically if no GPU is available)

```bash
pip install torch
```

## Data Setup

The OpenWebText-based scripts expect a local `openwebtext/` directory. Raw corpus files (`train_split.txt`, `val_split.txt`, and their `_v2` successors) and model checkpoints (`*.pkl`) are **not** tracked in this repo — see `.gitignore` — because they're tens of GB / hundreds of MB. To regenerate them:

```
openwebtext/
├── vocab_clean.txt     # cleaned character vocabulary (tracked, small)
├── train_split_v2.txt  # training corpus (regenerate with data-extract-v2.py)
└── val_split_v2.txt    # validation corpus (regenerate with data-extract-v2.py)
```

1. Point `data-extract-v2.py` at your local OpenWebText `.xz` archive folder and run it to build `train_split_v2.txt` / `val_split_v2.txt`.
2. Run `build_vocab.py` to derive `vocab_clean.txt` from the corpus.
3. Run `train_v2.py` to train.

The `llm.ipynb` prototype only needs `dracula.txt` in the working directory.

## Usage

### 1. Prototype on a small dataset
Run through `llm.ipynb` to train the bigram baseline model on `dracula.txt` and confirm the training/generation loop works end-to-end.

### 2. Train the full GPT model

```bash
python train_v2.py
```

Trains a fresh model (vocab/context-size changes mean checkpoints aren't compatible across versions), prints periodic train/val loss with elapsed time and current LR, and checkpoints to `model-0N.pkl` (latest) and `model-0N-best.pkl` (lowest val loss) as it goes.

### 3. Evaluate a checkpoint

```bash
python evaluate_v2.py
```

Reports parameter breakdown, validation loss over 200 random batches (nats + bits/char), and sample generations at multiple temperatures.

### 4. Chat with the trained model

```bash
python chatbox.py
```

Loads the current best checkpoint and drops into a loop where you type a prompt and the model streams back a completion.

## Known Limitations / Ideas for Improvement

- **Character-level tokenization** is still simple relative to BPE/subword tokenization (e.g. `tiktoken`) — would shorten sequences and likely improve quality further, at the cost of moving away from the "from scratch, character-level" design of this project.
- **No KV-cache in `generate()`** — every generation step recomputes the full forward pass, making sampling much slower than necessary.
- **No requirements.txt** — dependencies are limited to PyTorch, but pinning a version would help reproducibility.
- **Hardcoded paths** (`openwebtext/vocab_clean.txt`, etc.) assume a specific working directory.
