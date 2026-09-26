import mmap
import random
from collections import Counter

TRAIN_FILE = "openwebtext/train_split_v2.txt"
OUT_FILE = "openwebtext/vocab_clean.txt"

NUM_CHUNKS = 300
CHUNK_BYTES = 1_000_000  # 1MB per chunk -> ~300MB sampled total
COVERAGE = 0.9998        # keep chars covering this much of sampled mass
UNK_CHAR = '\x00'        # reserved fallback token for anything not in the clean vocab
DOC_BOUNDARY = '\x01'    # reserved doc-boundary token written by data-extract-v2.py

random.seed(1234)

counter = Counter()

with open(TRAIN_FILE, 'rb') as f:
    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        file_size = len(mm)
        print(f"{TRAIN_FILE} size: {file_size:,} bytes")
        for i in range(NUM_CHUNKS):
            start = random.randint(0, file_size - CHUNK_BYTES)
            mm.seek(start)
            block = mm.read(CHUNK_BYTES)
            text = block.decode('utf-8', errors='ignore').replace('\r', ' ')
            counter.update(text)
            if (i + 1) % 50 == 0:
                print(f"sampled {i+1}/{NUM_CHUNKS} chunks, unique chars so far: {len(counter)}", flush=True)

total_chars = sum(counter.values())
print(f"\ntotal sampled characters: {total_chars:,}")
print(f"unique characters seen in sample: {len(counter)}")

ranked = counter.most_common()
cumulative = 0
kept = []
for ch, cnt in ranked:
    cumulative += cnt
    kept.append(ch)
    if cumulative / total_chars >= COVERAGE:
        break

dropped = len(ranked) - len(kept)
print(f"keeping {len(kept)} characters covering {cumulative/total_chars:.5%} of sampled mass")
print(f"dropping {dropped} rare/noise characters")

vocab_set = set(kept)
vocab_set.add(' ')
vocab_set.add('\n')
vocab_set.add(UNK_CHAR)
vocab_set.add(DOC_BOUNDARY)

vocab_chars = sorted(vocab_set)
print(f"final vocab size: {len(vocab_chars)}")

with open(OUT_FILE, 'w', encoding='utf-8', newline='\n') as f:
    for ch in vocab_chars:
        f.write(ch + '\n')

print(f"wrote {OUT_FILE}")

sample = ''.join(c for c in vocab_chars if c.isprintable() and c not in (' ', '\n'))[:200]
print("sample of kept chars:", repr(sample))
