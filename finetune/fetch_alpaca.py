import json
import random
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
OUT = Path(__file__).parent.parent / "alpaca_data.jsonl"
SAMPLE_SIZE = 600
SEED = 42

print(f"Downloading Alpaca data...")
with urllib.request.urlopen(URL) as r:
    raw = json.loads(r.read())

print(f"Downloaded {len(raw)} examples")

examples = []
for item in raw:
    instruction = item["instruction"].strip()
    inp = item["input"].strip()
    output = item["output"].strip()

    if not instruction or not output:
        continue

    user_content = f"{instruction}\n\n{inp}" if inp else instruction
    examples.append({
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    })

random.seed(SEED)
random.shuffle(examples)
examples = examples[:SAMPLE_SIZE]

with open(OUT, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Saved {len(examples)} Alpaca examples to {OUT}")
