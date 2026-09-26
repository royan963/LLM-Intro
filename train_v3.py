import math
import mmap
import pickle
import random
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device, flush=True)

# ---------------------------------------------------------------------------
# Hyperparameters (same as train_v2.py -- only the corpus changed, to isolate
# the effect of the tar-parsing data fix)
# ---------------------------------------------------------------------------
batch_size = 64
block_size = 256
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2

max_lr = 3e-4
min_lr = 3e-5
warmup_iters = 200
grad_clip = 1.0

time_budget_hours = 4.0
time_budget_seconds = time_budget_hours * 3600

eval_interval = 250
eval_iters = 50

checkpoint_path = 'model-03.pkl'
best_checkpoint_path = 'model-03-best.pkl'

VOCAB_FILE = 'openwebtext/vocab_clean.txt'
UNK_CHAR = '\x00'

print(f"block_size={block_size} batch_size={batch_size} n_embd={n_embd} n_layer={n_layer} n_head={n_head}", flush=True)
print(f"max_lr={max_lr} min_lr={min_lr} warmup_iters={warmup_iters} grad_clip={grad_clip}", flush=True)
print(f"time_budget_hours={time_budget_hours} eval_interval={eval_interval} eval_iters={eval_iters}", flush=True)

# ---------------------------------------------------------------------------
# Vocab
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

# ---------------------------------------------------------------------------
# Data loading -- mmap opened once and reused for the whole run
# ---------------------------------------------------------------------------
def open_mmap(path):
    f = open(path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return f, mm

train_file, train_mm = open_mmap('openwebtext/train_split_v2.txt')
val_file, val_mm = open_mmap('openwebtext/val_split_v2.txt')


def get_random_chunk(split):
    mm = train_mm if split == 'train' else val_mm
    file_size = len(mm)
    start_pos = random.randint(0, file_size - block_size * batch_size)
    mm.seek(start_pos)
    block = mm.read(block_size * batch_size - 1)
    decoded_block = block.decode('utf-8', errors='ignore').replace('\r', ' ')
    data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data


def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Model (same architecture as v1/v2, parameterized by the globals above)
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
        out = torch.cat([h(x) for h in self.heads], dim=-1)
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
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
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
            loss = F.cross_entropy(logits, targets)

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


# ---------------------------------------------------------------------------
# Train from scratch on the corrected (tar-parsed) corpus
# ---------------------------------------------------------------------------
model = GPTLLM(vocab_size).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"model parameters: {n_params:,}", flush=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)

start_time = time.time()
best_val = float('inf')
it = 0

try:
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_budget_seconds:
            print(f"time budget of {time_budget_hours}h reached at step {it}, stopping", flush=True)
            break

        lr = get_lr(it, elapsed)
        for g in optimizer.param_groups:
            g['lr'] = lr

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if it % eval_interval == 0:
            losses = estimate_loss()
            elapsed_min = elapsed / 60
            print(f"step {it:>7} | {elapsed_min:7.1f} min | lr {lr:.2e} | "
                  f"train {losses['train']:.4f} | val {losses['val']:.4f}", flush=True)

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(model, f)

            if losses['val'] < best_val:
                best_val = losses['val'].item()
                with open(best_checkpoint_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  new best val loss {best_val:.4f} -> saved {best_checkpoint_path}", flush=True)

        it += 1

finally:
    final_losses = estimate_loss()
    print(f"FINAL step {it} | train {final_losses['train']:.4f} | val {final_losses['val']:.4f}", flush=True)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(model, f)
    if final_losses['val'] < best_val:
        best_val = final_losses['val'].item()
        with open(best_checkpoint_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"final model is new best ({best_val:.4f}) -> saved {best_checkpoint_path}", flush=True)
    print(f"best val loss over run: {best_val:.4f}", flush=True)
    print("TRAINING COMPLETE", flush=True)
