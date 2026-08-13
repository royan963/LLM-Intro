# Character-Level GPT from Scratch

A PyTorch implementation of a decoder-only, character-level GPT language model, built from the ground up (custom self-attention, multi-head attention, and transformer blocks — no `nn.Transformer` or Hugging Face). Includes a small proof-of-concept trained on *Dracula* and a larger version trained on OpenWebText, plus an interactive chat script for sampling from the trained model.

## Project Structure

| File | Description |
|---|---|
| `llm.ipynb` | Starter notebook. Trains a simple bigram language model on `dracula.txt` to validate the training loop and data pipeline before scaling up. |
| `gpt-v1.ipynb` | Full GPT model (self-attention, multi-head attention, transformer blocks) trained on the OpenWebText corpus. Notebook version of the main model. |
| `training.py` | Script version of the GPT training loop. Loads an existing checkpoint (`model-01.pkl`), continues training on OpenWebText, and saves the updated checkpoint. |
| `chatbox.py` | Interactive command-line chat interface. Loads `model-01.pkl` and generates text completions from user prompts. |
| `torch-examples.ipynb` | Scratch notebook of PyTorch fundamentals (tensor ops, embeddings, softmax, matrix multiplication, CPU vs. GPU benchmarking) used while learning the building blocks for the model. |
| `dracula.txt` | Small text corpus (public-domain novel) used for the early bigram prototype. |
| `model-01.pkl` | Pickled, trained model checkpoint (`GPTLLM` instance with learned weights). |

## Model Architecture

`GPTLLM` is a decoder-only transformer with:

- **Token + positional embeddings** — learned embeddings for each character and each position in the sequence.
- **Stacked transformer blocks** (`n_layer = 8`), each containing:
  - Multi-head causal self-attention (`n_head = 8`), with a lower-triangular mask so each token only attends to previous tokens.
  - A position-wise feed-forward network (4x expansion, ReLU, projection back down).
  - Residual connections and layer normalization around both sub-layers.
- **Final layer norm** + a linear head projecting to vocabulary logits.
- **Dropout** (`0.2`) applied in attention and feed-forward layers for regularization.

### Hyperparameters (`gpt-v1.ipynb` / `training.py`)

| Parameter | Value |
|---|---|
| `batch_size` | 64 |
| `block_size` (context length) | 128 |
| `n_embd` (embedding dim) | 384 |
| `n_layer` | 8 |
| `n_head` | 8 |
| `dropout` | 0.2 |
| `learning_rate` | 3e-4 |
| `max_iters` | 5000 |
| `eval_iters` | 500 |

Tokenization is **character-level**: the vocabulary is built directly from the unique characters in the training corpus, with simple `encode`/`decode` lookup dictionaries (no BPE/subword tokenization).

## Requirements

- Python 3.9+
- PyTorch (with CUDA support recommended — training falls back to CPU automatically if no GPU is available)

```bash
pip install torch
```

## Data Setup

The OpenWebText-based scripts (`gpt-v1.ipynb`, `training.py`, `chatbox.py`) expect a local `openwebtext/` directory containing:

```
openwebtext/
├── vocab.txt        # defines the character vocabulary
├── train_split.txt  # training corpus
└── val_split.txt    # validation corpus
```

`training.py` reads random chunks from these files via memory-mapped I/O (`mmap`) rather than loading the full corpus into memory, so it can scale to large text files efficiently.

The `llm.ipynb` prototype only needs `dracula.txt` in the working directory.

## Usage

### 1. Prototype on a small dataset
Run through `llm.ipynb` to train the bigram baseline model on `dracula.txt` and confirm the training/generation loop works end-to-end.

### 2. Train the full GPT model
Either run `gpt-v1.ipynb` cell-by-cell, or run the script version:

```bash
python training.py
```

This loads the existing checkpoint (`model-01.pkl`), continues training for `max_iters` steps on OpenWebText, prints periodic train/val loss, and saves the updated weights back to `model-01.pkl`.

> Note: both `gpt-v1.ipynb` and `training.py` currently load `model-01.pkl` at start-up rather than initializing fresh weights — to train a brand-new model from scratch, skip the `pickle.load` step and use the freshly constructed `GPTLLM(vocab_size)` instance instead.

### 3. Chat with the trained model

```bash
python chatbox.py
```

This loads `model-01.pkl` and drops into a loop where you type a prompt and the model streams back a 500-token completion.

## Known Limitations / Ideas for Improvement

- **Character-level tokenization** is simple but inefficient — moving to BPE/subword tokenization (e.g. `tiktoken`) would shorten sequences and likely improve quality.
- **No validation-based early stopping or checkpoint versioning** — every training run overwrites `model-01.pkl` in place.
- **`training.py` / `gpt-v1.ipynb` always load a checkpoint before training**, so there's no clean "train from scratch" entry point without manually editing the code.
- **No requirements.txt** — dependencies are limited to PyTorch, but pinning a version would help reproducibility.
- **Hardcoded paths** (`openwebtext/vocab.txt`, etc.) assume a specific working directory.
