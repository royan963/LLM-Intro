import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

batch_size = 64
block_size = 256          # matches train_v2.py
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2

MODEL_FILE = 'model-02-best.pkl'
VOCAB_FILE = 'openwebtext/vocab_clean.txt'
UNK_CHAR = '\x00'

with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))

vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
unk_id = string_to_int[UNK_CHAR]
encode = lambda s: [string_to_int.get(c, unk_id) for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])


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

    def generate(self, index, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, loss = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)

        return index


# ---------------------------------------------------------------------------
print('loading model parameters...')
with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print('loaded successfully')
model = model.to(device)
model.eval()


def count_params(module):
    return sum(p.numel() for p in module.parameters())


total_params = count_params(model)
tok_emb_params = count_params(model.token_embedding_table)
pos_emb_params = count_params(model.position_embedding_table)
lm_head_params = count_params(model.lm_head)
blocks_params = count_params(model.blocks)

print("\n===== PARAMETER COUNTS (model-02, cleaned vocab) =====")
print(f"vocab_size: {vocab_size}")
print(f"Total parameters: {total_params:,}")
print(f"  token_embedding_table: {tok_emb_params:,}")
print(f"  position_embedding_table: {pos_emb_params:,}")
print(f"  lm_head: {lm_head_params:,}")
print(f"  embedding+lm_head combined: {tok_emb_params + lm_head_params:,}")
print(f"  transformer blocks (all {n_layer} layers): {blocks_params:,}")


def get_random_chunk(split, size_bytes):
    filename = "openwebtext/train_split.txt" if split == 'train' else "openwebtext/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, file_size - size_bytes)
            mm.seek(start_pos)
            block = mm.read(size_bytes - 1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', ' ')
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            return data


def get_val_batch():
    data = get_random_chunk('val', block_size * batch_size)
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


print("\n===== VALIDATION LOSS (200 random batches, independent offsets) =====")
random.seed()
num_batches = 200
losses = torch.zeros(num_batches)
with torch.no_grad():
    for k in range(num_batches):
        X, Y = get_val_batch()
        logits, loss = model(X, Y)
        losses[k] = loss.item()

mean_loss = losses.mean().item()
std_loss = losses.std().item()
bits_per_char = mean_loss / torch.log(torch.tensor(2.0)).item()

print(f"Mean val loss: {mean_loss:.4f} nats  (std across batches: {std_loss:.4f})")
print(f"Bits per character: {bits_per_char:.4f}")

print("\n===== GENERATION SAMPLES =====")
prompts = ["The ", "In 2019, ", "Scientists"]
settings = [(0.8, 50), (1.0, 50)]

with torch.no_grad():
    for prompt in prompts:
        for temperature, top_k in settings:
            context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            out = model.generate(context, max_new_tokens=400, temperature=temperature, top_k=top_k)
            generated = decode(out[0].tolist())
            print(f"\n--- prompt={prompt!r} temperature={temperature} top_k={top_k} ---")
            print(generated)

print("\n===== DONE =====")
