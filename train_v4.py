import math
import pickle
import queue
import random
import threading
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------------------------------------------------------------------------
# Hyperparameters (same as train_v3.py except dropout -- see notes below)
#
# Changes vs v3:
#   1. Batches draw each of the 64 sequences from an independent random offset
#      in the corpus (v3 took all 64 from one contiguous 16KB chunk, so a batch
#      was really only 2-4 overlapping documents).
#   2. dropout 0.2 -> 0.0: the run sees ~3% of one epoch, so there is nothing
#      to overfit and dropout only slows learning.
#   3. Speed: attention heads are computed in one fused call
#      (F.scaled_dot_product_attention), bf16 autocast, TF32 matmuls,
#      torch.compile (needs triton / triton-windows), and background threads
#      prepare batches while the GPU trains.
#
# The parameter layout and the math are unchanged (still post-LN, same
# Head/MultiHeadAttention/FeedFoward/Block/GPTLLM modules and names), so the
# pickled checkpoint loads in chatbox.py / evaluate_v3.py / export_onnx.py.
# ---------------------------------------------------------------------------
batch_size = 64
block_size = 256
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.0

max_lr = 3e-4
min_lr = 3e-5
warmup_iters = 200
grad_clip = 1.0

time_budget_hours = 4.0
time_budget_seconds = time_budget_hours * 3600

eval_interval = 500
eval_iters = 100

checkpoint_path = 'model-04.pkl'
best_checkpoint_path = 'model-04-best.pkl'

VOCAB_FILE = 'openwebtext/vocab_clean.txt'
UNK_CHAR = '\x00'

print(f"block_size={block_size} batch_size={batch_size} n_embd={n_embd} n_layer={n_layer} n_head={n_head} dropout={dropout}", flush=True)
print(f"max_lr={max_lr} min_lr={min_lr} warmup_iters={warmup_iters} grad_clip={grad_clip}", flush=True)
print(f"time_budget_hours={time_budget_hours} eval_interval={eval_interval} eval_iters={eval_iters}", flush=True)

# ---------------------------------------------------------------------------
# Vocab -- plus a codepoint -> id lookup table so encoding is a numpy gather
# instead of a Python dict lookup per character
# ---------------------------------------------------------------------------
with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))

vocab_size = len(chars)
print(f"vocab_size: {vocab_size}", flush=True)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
unk_id = string_to_int[UNK_CHAR]

encode = lambda s: [string_to_int.get(c, unk_id) for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

codepoint_to_id = np.full(0x110000, unk_id, dtype=np.int64)
for ch, i in string_to_int.items():
    codepoint_to_id[ord(ch)] = i
codepoint_to_id[ord('\r')] = string_to_int[' ']  # v3 replaced '\r' with ' '


def encode_np(s):
    codepoints = np.frombuffer(s.encode('utf-32-le'), dtype=np.uint32)
    return codepoint_to_id[codepoints]


# ---------------------------------------------------------------------------
# Data loading -- every sequence in a batch comes from its own random offset
# ---------------------------------------------------------------------------
SPLIT_FILES = {
    'train': 'openwebtext/train_split_v2.txt',
    'val': 'openwebtext/val_split_v2.txt',
}

# worst case is 4 bytes per character, so this always holds block_size + 1
# characters unless the window is mostly undecodable
read_bytes = 4 * (block_size + 1) + 8


class CorpusReader:
    """One open file handle per split. Plain file reads release the GIL, so
    several reader threads can wait on the disk in parallel."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.files = {split: open(path, 'rb') for split, path in SPLIT_FILES.items()}
        self.sizes = {split: f.seek(0, 2) for split, f in self.files.items()}

    def get_sequence(self, split):
        f = self.files[split]
        while True:
            f.seek(self.rng.randint(0, self.sizes[split] - read_bytes))
            s = f.read(read_bytes).decode('utf-8', errors='ignore')
            if len(s) >= block_size + 1:
                return encode_np(s[:block_size + 1])

    def make_batch(self, split):
        data = np.stack([self.get_sequence(split) for _ in range(batch_size)])
        data = torch.from_numpy(data)
        x = data[:, :-1].contiguous()
        y = data[:, 1:].contiguous()
        if device == 'cuda':
            x, y = x.pin_memory(), y.pin_memory()
        return x, y


def to_device(x, y):
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


class BatchPrefetcher:
    """Builds training batches on background threads so the GPU never waits."""

    def __init__(self, split, workers=4, depth=32):
        self.split = split
        self.q = queue.Queue(maxsize=depth)
        for _ in range(workers):
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        reader = CorpusReader()
        while True:
            self.q.put(reader.make_batch(self.split))

    def next(self):
        return to_device(*self.q.get())


eval_reader = CorpusReader()


def get_batch(split):
    return to_device(*eval_reader.make_batch(split))


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Model -- same modules and parameters as v1/v2/v3. Only MultiHeadAttention's
# forward changes: it stacks the per-head key/query/value weights and runs all
# heads in a single fused causal attention call, which computes exactly what
# the per-head loop does.
# ---------------------------------------------------------------------------
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
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        H = len(self.heads)
        # (3 * H * hs, C): all query weights, then all key weights, then all value weights
        w = torch.cat([h.query.weight for h in self.heads] +
                      [h.key.weight for h in self.heads] +
                      [h.value.weight for h in self.heads], dim=0)
        q, k, v = F.linear(x, w).split(C, dim=-1)
        q = q.view(B, T, H, -1).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, H, -1).transpose(1, 2)
        v = v.view(B, T, H, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, C)  # heads concatenated, same order as torch.cat
        out = self.dropout(self.proj(out))
        return out


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
        x = self.ln2(x + y)
        return x


class GPTLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits.float(), targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index


# ---------------------------------------------------------------------------
# LR schedule: linear warmup by iteration count, then cosine decay driven by
# wall-clock progress through the time budget (robust to unknown steps/sec)
# ---------------------------------------------------------------------------
def get_lr(it, elapsed_seconds):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    progress = min(elapsed_seconds / time_budget_seconds, 1.0)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def save(path):
    # pickle the uncompiled module so chatbox.py etc. can load it as before
    with open(path, 'wb') as f:
        pickle.dump(raw_model, f)


# ---------------------------------------------------------------------------
# Train from scratch on the corrected (tar-parsed) corpus
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    raw_model = GPTLLM(vocab_size).to(device)
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f"model parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=max_lr, fused=(device == 'cuda'))
    model = torch.compile(raw_model) if device == 'cuda' else raw_model
    prefetcher = BatchPrefetcher('train')

    start_time = time.time()
    best_val = float('inf')
    it = 0
    chars_seen = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > time_budget_seconds:
                print(f"time budget of {time_budget_hours}h reached at step {it}, stopping", flush=True)
                break

            lr = get_lr(it, elapsed)
            for g in optimizer.param_groups:
                g['lr'] = lr

            xb, yb = prefetcher.next()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            chars_seen += batch_size * block_size

            if it % eval_interval == 0:
                losses = estimate_loss()
                elapsed_min = elapsed / 60
                steps_per_sec = it / elapsed if elapsed > 0 else 0.0
                print(f"step {it:>7} | {elapsed_min:7.1f} min | {steps_per_sec:5.2f} it/s | "
                      f"{chars_seen / 1e9:6.3f}B chars | lr {lr:.2e} | "
                      f"train {losses['train']:.4f} | val {losses['val']:.4f}", flush=True)

                save(checkpoint_path)

                if losses['val'] < best_val:
                    best_val = losses['val'].item()
                    save(best_checkpoint_path)
                    print(f"  new best val loss {best_val:.4f} -> saved {best_checkpoint_path}", flush=True)

            it += 1

    finally:
        final_losses = estimate_loss()
        print(f"FINAL step {it} | {chars_seen / 1e9:.3f}B chars | "
              f"train {final_losses['train']:.4f} | val {final_losses['val']:.4f}", flush=True)
        save(checkpoint_path)
        if final_losses['val'] < best_val:
            best_val = final_losses['val'].item()
            save(best_checkpoint_path)
            print(f"final model is new best ({best_val:.4f}) -> saved {best_checkpoint_path}", flush=True)
        print(f"best val loss over run: {best_val:.4f}", flush=True)
        print("TRAINING COMPLETE", flush=True)
