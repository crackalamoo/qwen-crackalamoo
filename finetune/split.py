import json
import random
from pathlib import Path
from mlx_lm import load

SRC = Path(__file__).parent.parent / "finetune_data.jsonl"
BLOG_SRC = Path(__file__).parent.parent / "blog_data.jsonl"
TRAIN = Path(__file__).parent.parent / "train.jsonl"
VAL = Path(__file__).parent.parent / "val.jsonl"
VAL_FRACTION = 0.05
SEED = 42
MIN_COMPLETION_TOKENS = 20

_, tokenizer = load("mlx-community/Qwen3-8B-4bit")

with open(SRC) as f:
    raw = [json.loads(l) for l in f]

# blog examples skip the token filter — they're already long-form
blog = []
if BLOG_SRC.exists():
    with open(BLOG_SRC) as f:
        blog = [json.loads(l) for l in f]

discord = [
    ex for ex in raw
    if len(tokenizer.encode(ex["messages"][-1]["content"])) >= MIN_COMPLETION_TOKENS
]
print(f"Discord filtered: {len(raw)} → {len(discord)} ({100*len(discord)/len(raw):.1f}% kept)")
print(f"Blog examples: {len(blog)}")
examples = discord + (blog * 5)

random.seed(SEED)
random.shuffle(examples)

split = int(len(examples) * (1 - VAL_FRACTION))
train, val = examples[:split], examples[split:]

with open(TRAIN, "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")

with open(VAL, "w") as f:
    for ex in val:
        f.write(json.dumps(ex) + "\n")

print(f"train: {len(train)}, val: {len(val)}")

