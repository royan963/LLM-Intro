"""
Export the trained GPT (model-04-best.pkl) so it can run in a web browser.

Run from the llm-intro folder (same place you run chatbox.py):
    pip install onnx onnxruntime
    python export_onnx.py

Writes, into ./web-model/:
    model.onnx   int8-quantized model (~15 MB) that the portfolio site loads
    vocab.json   the character list, in the same order the model was trained with
Copy both files into the site's  model/  folder.
"""
import io
import json
import os
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F

# Same hyperparameters as train_v4.py / chatbox.py (dropout is ignored in eval mode)
block_size = 256
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2
device = 'cpu'

MODEL_PATH = 'model-04-best.pkl'
VOCAB_PATH = 'openwebtext/vocab_clean.txt'
OUT_DIR = 'web-model'


# ---- model classes: copied from chatbox.py so pickle can find them ----
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        return self.ln2(x + y)


class GPTLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(self.ln_f(x))
        return logits, None


class NextCharLogits(nn.Module):
    """What the browser needs: (1, T) character ids -> logits for the next character."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, index):
        logits, _ = self.model(index)
        return logits[:, -1, :]


def load_checkpoint(path):
    # The checkpoint was pickled on a GPU machine; load its tensors onto the CPU.
    orig = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    finally:
        torch.storage._load_from_bytes = orig


def main():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        chars = sorted(set(f.read()))   # same ordering chatbox.py uses

    model = load_checkpoint(MODEL_PATH).cpu().eval()
    assert model.lm_head.out_features == len(chars), \
        f'vocab mismatch: model has {model.lm_head.out_features} outputs, vocab file has {len(chars)} chars'

    os.makedirs(OUT_DIR, exist_ok=True)
    fp32_path = os.path.join(OUT_DIR, 'model-fp32.onnx')
    out_path = os.path.join(OUT_DIR, 'model.onnx')

    wrapper = NextCharLogits(model).eval()
    example = torch.randint(0, len(chars), (1, 32))
    torch.onnx.export(
        wrapper, (example,), fp32_path,
        input_names=['index'], output_names=['logits'],
        dynamic_axes={'index': {1: 'T'}},
        opset_version=17, dynamo=False,
    )

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(fp32_path, out_path, weight_type=QuantType.QInt8)
    os.remove(fp32_path)

    with open(os.path.join(OUT_DIR, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump({'chars': chars, 'block_size': block_size}, f, ensure_ascii=False)

    # Sanity check: the browser model should pick the same likely next characters as PyTorch.
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path)
    stoi = {c: i for i, c in enumerate(chars)}
    unk = stoi.get('\x00', 0)
    prompt = 'The president said that '
    ids = torch.tensor([[stoi.get(c, unk) for c in prompt]])
    with torch.no_grad():
        ref = wrapper(ids).numpy()[0]
    got = sess.run(None, {'index': ids.numpy().astype(np.int64)})[0][0]
    top_ref = [chars[i] for i in np.argsort(-ref)[:5]]
    top_got = [chars[i] for i in np.argsort(-got)[:5]]
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'wrote {out_path} ({size_mb:.1f} MB) and vocab.json ({len(chars)} chars)')
    print(f'top-5 next chars after {prompt!r}: pytorch={top_ref} onnx={top_got}')


if __name__ == '__main__':
    main()
