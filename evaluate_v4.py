import pickle

import torch
from torch.nn import functional as F

# Model classes must live in __main__ for pickle to find them. The v4 classes
# have the same parameters and compute the same function as the v1-v3 ones,
# so they load every checkpoint from model-03 onward.
from train_v4 import (Head, MultiHeadAttention, FeedFoward, Block, GPTLLM,  # noqa: F401
                      CorpusReader, block_size, batch_size, device, encode, decode, vocab_size)

MODEL_FILES = {
    'model-03-best': 'model-03-best.pkl',
    'model-04-best': 'model-04-best.pkl',
}
NUM_BATCHES = 200
EVAL_SEED = 1234
SAMPLE_SEED = 42


def load(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model.to(device).eval()


@torch.no_grad()
def sample(model, prompt, max_new_tokens=400, temperature=1.0, top_k=None):
    index = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(index[:, -block_size:])
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        index = torch.cat((index, torch.multinomial(probs, num_samples=1)), dim=1)
    return decode(index[0].tolist())


# Every model is scored on the exact same batches (each sequence from its own
# random offset in the validation split), so the numbers compare directly.
reader = CorpusReader(seed=EVAL_SEED)
batches = [reader.make_batch('val') for _ in range(NUM_BATCHES)]

models = {name: load(path) for name, path in MODEL_FILES.items()}

print(f"\n===== VALIDATION LOSS ({NUM_BATCHES} batches x {batch_size} independent sequences, "
      f"same batches for every model, fp32) =====")
print(f"vocab_size: {vocab_size}")
results = {}
with torch.no_grad():
    for name, model in models.items():
        losses = torch.zeros(NUM_BATCHES)
        for k, (x, y) in enumerate(batches):
            _, loss = model(x.to(device), y.to(device))
            losses[k] = loss.item()
        results[name] = losses
        mean = losses.mean().item()
        stderr = losses.std().item() / NUM_BATCHES ** 0.5
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {n_params:,} params | val loss {mean:.4f} nats (+/- {stderr:.4f} s.e.) | "
              f"{mean / torch.log(torch.tensor(2.0)).item():.4f} bits/char")

names = list(results)
diff = results[names[1]] - results[names[0]]
print(f"{names[1]} - {names[0]}: {diff.mean().item():+.4f} nats per char "
      f"(paired s.e. {diff.std().item() / NUM_BATCHES ** 0.5:.4f}, "
      f"{names[1]} better on {(diff < 0).sum().item()}/{NUM_BATCHES} batches)")

print("\n===== GENERATION SAMPLES (same random seed for each model) =====")
prompts = ["The ", "In 2019, ", "Scientists"]
settings = [(0.8, 50), (1.0, 50)]
for prompt in prompts:
    for temperature, top_k in settings:
        for name, model in models.items():
            torch.manual_seed(SAMPLE_SEED)
            print(f"\n--- {name} | prompt={prompt!r} temperature={temperature} top_k={top_k} ---")
            print(sample(model, prompt, temperature=temperature, top_k=top_k))

print("\n===== DONE =====")
