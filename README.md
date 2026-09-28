# Character-Level GPT from Scratch

A PyTorch implementation of a decoder-only, character-level GPT language model, built from the ground up (custom self-attention, multi-head attention, and transformer blocks — no `nn.Transformer` or Hugging Face). Includes a small proof-of-concept trained on *Dracula* and a larger version trained on OpenWebText, plus an interactive chat script for sampling from the trained model.

## Who Built What

This project was built in two phases: I wrote the original model and pipeline myself while learning, and later used [Claude Code](https://claude.com/claude-code) (Anthropic's AI coding assistant) to evaluate it, find bugs, and improve it. Commits made with Claude's help are marked `Co-Authored-By: Claude` in the git history.

### What I built myself

- **The GPT model from scratch**: `Head` (self-attention with a causal mask), `MultiHeadAttention`, `FeedForward`, `Block` (transformer block with residuals and layer norm), and the full `GPTLLM` model with token and positional embeddings and `generate()`.
- **`torch-examples.ipynb`**: my scratch notebook for learning PyTorch basics (tensors, embeddings, softmax, matmul, CPU vs. GPU).
- **`llm.ipynb`**: the bigram prototype trained on `dracula.txt` to get the training loop working.
- **`gpt-v1.ipynb`**: the full GPT notebook trained on OpenWebText.
- **`training.py`**: the original training script (memory-mapped random batch sampling, loss estimation, AdamW training loop, pickled checkpoints).
- **`openwebtext/data-extract.py`**: the original OpenWebText extraction and train/val split.
- **`chatbox.py`**: the interactive chat interface.
- **`model-01.pkl`**: the first trained checkpoint (53M params).
- The overall project design: a character-level, from-scratch GPT with no `nn.Transformer` or Hugging Face.

### What Claude helped me do

- **Evaluating my model**: wrote `evaluate.py`, `evaluate_v2.py` and `evaluate_v3.py` (parameter breakdown, validation loss over 200 batches, bits/char, sample generations).
- **Finding the vocab problem**: noticed that 46.6% of `model-01`'s parameters went to a 32k-character vocab that was mostly Unicode noise, and wrote `build_vocab.py` to keep only the frequent characters.
- **Improving training**: wrote `train_v2.py` / `train_v3.py`, which add a longer context (`block_size` 128 → 256), LR warmup + cosine decay, gradient clipping, and time-boxed training with checkpointing. The model architecture stayed mine.
- **Finding the tar-parsing bug** in my `data-extract.py` (the `.xz` files are tar archives, so the corpus was full of tar headers and NUL bytes) and writing the fixed `data-extract-v2.py`.
- **Running the retraining** that produced `model-02-best.pkl`, `model-03-best.pkl` and `model-04-best.pkl`.
- **Faster, better-sampled training (`train_v4.py`)**: each sequence in a batch now comes from its own random place in the corpus (v3 took all 64 from one 16 KB chunk), dropout is off (the run never sees the same text twice), and the attention heads run in one fused call with bf16 and `torch.compile`. About 4× more training steps in the same 4 hours, with the same parameters and math as my model, so the checkpoints still load in `chatbox.py`.
- **`evaluate_v4.py`**: scores several checkpoints on the exact same validation batches so they can be compared directly.
- **Small `chatbox.py` updates**: pointed it at the new vocab and checkpoints, added an unknown-character fallback so unseen characters don't crash it, fixed `decode` joining characters with spaces, and switched generation to `eval()` / `no_grad()`.
- **`export_onnx.py`**: exports the current best checkpoint to a quantized ONNX model (`web-model/`) so it can run in a browser.
- **Git cleanup**: merging the remote history and flattening a duplicate nested folder.
- **Writing most of this README**, including the evaluation write-up and results table below.

## Project Structure

| File | Description |
|---|---|
| `llm.ipynb` | Starter notebook. Trains a simple bigram language model on `dracula.txt` to validate the training loop and data pipeline before scaling up. |
| `gpt-v1.ipynb` | Full GPT model (self-attention, multi-head attention, transformer blocks) trained on the OpenWebText corpus. Notebook version of the main model. |
| `training.py` | Original script version of the GPT training loop. Loads `model-01.pkl` and continues training with a flat learning rate. Kept for reference; `train_v4.py` supersedes it. |
| `train_v2.py` | Improved training script: fresh model on the cleaned vocab, `block_size=256`, warmup + cosine LR decay, gradient clipping, time-boxed with periodic checkpointing. Trained on the (still tar-corrupted) v1 corpus. |
| `train_v3.py` | Same architecture/schedule as `train_v2.py`, retrained on the corrected (tar-parsed) corpus. |
| `train_v4.py` | Same model and schedule as `train_v3.py`, with independently sampled batches, no dropout, and faster training (fused attention, bf16, `torch.compile`, background data loading). **Produces the current recommended checkpoint.** |
| `chatbox.py` | Interactive command-line chat interface. Loads the current best checkpoint and generates text completions from user prompts. |
| `evaluate.py` / `evaluate_v2.py` / `evaluate_v3.py` | Non-interactive evaluation: parameter counts, validation loss (nats + bits/char) averaged over many random batches, and sample generations at multiple temperatures. |
| `evaluate_v4.py` | Scores `model-03-best` and `model-04-best` on the same 200 validation batches (paired comparison) and prints side-by-side samples with a shared random seed. |
| `export_onnx.py` | Exports the current best checkpoint to an int8-quantized ONNX model in `web-model/` for running in a browser. |
| `build_vocab.py` | Builds a cleaned character vocabulary by sampling the corpus and keeping only characters above a frequency threshold (see *Known issues found & fixed* below). |
| `data-extract.py` | Original OpenWebText extraction script. **Has a bug** — see below. Kept for reference. |
| `data-extract-v2.py` | Corrected extraction script that properly parses the `.xz` archives as tar files instead of raw text. |
| `torch-examples.ipynb` | Scratch notebook of PyTorch fundamentals (tensor ops, embeddings, softmax, matrix multiplication, CPU vs. GPU benchmarking) used while learning the building blocks for the model. |
| `dracula.txt` | Small text corpus (public-domain novel) used for the early bigram prototype. |
| `model-01.pkl` | Original checkpoint (53M params, 32k-char vocab, trained on the tar-corrupted corpus). Not tracked in git (see `.gitignore`) — too large for GitHub. |
| `model-02-best.pkl` | Checkpoint from `train_v2.py` (14.5M params, 254-char vocab, `block_size=256`, still trained on the tar-corrupted corpus). Not tracked in git. **Not directly re-evaluable** — `vocab_clean.txt` was later regenerated for `model-03` and no longer matches its embedding indices; kept for historical reference only. |
| `model-03-best.pkl` | Checkpoint from `train_v3.py` — same architecture, trained on the corrected corpus (14.4M params, 199-char vocab, matches the current `vocab_clean.txt`). Not tracked in git. |
| `model-04-best.pkl` | Checkpoint from `train_v4.py` — same architecture and vocab as model-03, trained on 4× more text. **Use this one.** Not tracked in git. |

## Evaluation & fixes (this pass)

The original `model-01.pkl` was evaluated end-to-end (parameter breakdown, 200-batch validation loss, sample generations) and three issues were found and fixed:

1. **Bloated vocabulary.** `vocab.txt` was built by dumping every unique character seen across the entire 46GB corpus — 32,172 characters, mostly one-off Unicode noise. This meant **46.6% of the model's 53M parameters** were spent on the token embedding + output head alone. `build_vocab.py` now samples the corpus and keeps only characters covering 99.98% of observed frequency mass (254 characters), which cut total parameters to 14.5M while *improving* validation loss.
2. **Short context.** `block_size` was 128 characters (~20-25 words) — too short for topical coherence. Increased to 256 in `train_v2.py`.
3. **No LR schedule / short training.** The original loop used a flat `lr=3e-4` for a fixed 5000 iterations. `train_v2.py` adds linear warmup + cosine decay (decay driven by wall-clock progress through a time budget, so it's robust to unknown steps/sec) and gradient clipping, and trains until a time budget is reached (checkpointing periodically so no progress is lost).

A fourth, more serious issue was found while testing the retrained model:

4. **Corpus corruption from a tar-parsing bug.** The original OpenWebText `.xz` archives are actually **compressed tar files** (each contains hundreds of individual `<hash>.txt` documents). `data-extract.py` decoded the raw decompressed byte stream directly as UTF-8 text instead of parsing it as a tar archive, so `train_split.txt` / `val_split.txt` were contaminated throughout with tar headers and NUL padding (~11% of all bytes in a sampled region were literal `\x00`, plus tar entry names like `0999049-cf978a7f....txt` leaking in as if they were prose). `data-extract-v2.py` fixes this by opening each archive with Python's `tarfile` module and extracting only the genuine per-document text, separated by an explicit document-boundary character, producing `train_split_v2.txt` / `val_split_v2.txt` (0% NUL bytes, verified). `train_v3.py` retrains the same architecture from scratch on this corrected corpus.

| | model-01 (original) | model-02-best (vocab+context+schedule fix) | model-03-best (+ corrected corpus) | model-04-best (+ better batches, no dropout, faster training) |
|---|---|---|---|---|
| Parameters | 53,163,180 | 14,480,894 | 14,438,599 | 14,438,599 |
| `vocab_size` | 32,172 | 254 | 199 | 199 |
| `block_size` | 128 | 256 | 256 | 256 |
| Characters trained on | — | — | ~1.1B | ~4.4B |
| Val loss | 1.3095 nats | 1.0182 nats | 1.1504 nats (1.1459\*) | 1.0233 nats\* |
| Bits/char | 1.889 | 1.469 | 1.660 (1.653\*) | 1.476\* |

\* From `evaluate_v4.py`, which scores model-03 and model-04 on the *same* 200 validation batches. model-04 is better on all 200 of them, by 0.123 nats per character on average (paired standard error 0.0005).

**The val-loss numbers above are not directly comparable across corpora.** model-01 and model-02 were evaluated on the *tar-corrupted* corpus, which contains long, highly repetitive, trivially-predictable byte sequences (NUL padding, fixed permission strings, tar magic bytes) — the model can nail those almost perfectly, which pulls the average loss down without reflecting genuine language-modeling skill. model-03 is trained and evaluated entirely on clean, real text with no such shortcut, so its higher raw loss actually reflects a *harder, more honest* task, not a worse model. Judged on generation samples instead, model-03 is a clear improvement: it produces properly structured multi-sentence paragraphs (correct paragraph breaks, consistent news-article style, recurring named entities within a passage) versus the more fragmented, run-on output of model-02, though it's still not semantically coherent over long spans and occasionally invents garbled words.

A fifth round of fixes went into `train_v4.py`, with the model itself unchanged:

5. **Correlated batches.** v3 read one contiguous 16 KB chunk per step and took all 64 training sequences from inside it, so a "batch of 64" was really 2–4 overlapping documents. v4 gives every sequence its own random offset in the corpus.
6. **Unneeded dropout.** A 4-hour run sees only a few percent of the 36 GB corpus, so train and val loss stay equal and there is nothing to overfit. v4 sets dropout to 0.
7. **Slow training.** v4 runs all attention heads in one fused `F.scaled_dot_product_attention` call (same result as the per-head loop), trains in bf16 with `torch.compile`, and prepares batches on background threads. That took training from ~4.6 to ~18.6 steps/s, so the same 4 hours covers ~4.4B characters instead of ~1.1B.

model-04 reaches 1.476 bits/char against model-03's 1.653 on the same data. Its samples stay on topic for longer and handle quotations and dialogue better, though they still don't make sense beyond a sentence or so. Train and val loss were still equal and still falling when the time ran out, so a longer run would keep improving it.

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

| Parameter | `training.py` (original) | `train_v2.py` / `train_v3.py` | `train_v4.py` (current) |
|---|---|---|---|
| `batch_size` | 64 | 64 | 64 |
| `block_size` (context length) | 128 | 256 | 256 |
| `n_embd` (embedding dim) | 384 | 384 | 384 |
| `n_layer` | 8 | 8 | 8 |
| `n_head` | 8 | 8 | 8 |
| `dropout` | 0.2 | 0.2 | 0.0 |
| batch sampling | 64 windows from one chunk | 64 windows from one chunk | 64 independent random offsets |
| precision | fp32 | fp32 | bf16 autocast + `torch.compile` |
| `learning_rate` | flat 3e-4 | warmup to 3e-4, cosine decay to 3e-5 | warmup to 3e-4, cosine decay to 3e-5 |
| stopping condition | fixed 5000 iters | time-boxed (4h), checkpointed periodically | time-boxed (4h), checkpointed periodically |

`train_v3.py` is identical to `train_v2.py` except for which corpus it reads — it exists as a separate file so both training runs and their checkpoints stay independently reproducible.

Tokenization is **character-level**: `build_vocab.py` derives a frequency-cleaned vocabulary from the corpus, with simple `encode`/`decode` lookup dictionaries (no BPE/subword tokenization) plus a reserved UNK character for anything outside the vocab.

## Requirements

- Python 3.9+
- PyTorch (with CUDA support recommended — training falls back to CPU automatically if no GPU is available)
- NumPy
- Triton, for `torch.compile` in `train_v4.py` (on Windows, install the community `triton-windows` package)

```bash
pip install torch numpy
pip install triton-windows   # Windows only; on Linux, triton ships with torch
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
3. Run `train_v4.py` to train.

The `llm.ipynb` prototype only needs `dracula.txt` in the working directory.

## Usage

### 1. Prototype on a small dataset
Run through `llm.ipynb` to train the bigram baseline model on `dracula.txt` and confirm the training/generation loop works end-to-end.

### 2. Train the full GPT model

```bash
python train_v4.py
```

Trains a fresh model (vocab/context-size changes mean checkpoints aren't compatible across versions) on the corrected corpus, prints periodic train/val loss with elapsed time and current LR, and checkpoints to `model-0N.pkl` (latest) and `model-0N-best.pkl` (lowest val loss) as it goes.

### 3. Evaluate a checkpoint

```bash
python evaluate_v4.py
```

Scores `model-03-best` and `model-04-best` on the same 200 validation batches (nats + bits/char, paired difference) and prints side-by-side sample generations at multiple temperatures.

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
