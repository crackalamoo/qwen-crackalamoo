import json
import random
from pathlib import Path

SRC = Path(__file__).parent.parent / "finetune_data.jsonl"
TRAIN = Path(__file__).parent.parent / "train.jsonl"
VAL = Path(__file__).parent.parent / "val.jsonl"
VAL_FRACTION = 0.05
SEED = 42

with open(SRC) as f:
    examples = [json.loads(l) for l in f]

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

