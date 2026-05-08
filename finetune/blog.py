import json
import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "blog.json"
OUT = Path(__file__).parent.parent / "blog_data.jsonl"
PARAGRAPHS_PER_CHUNK = 3
PUNCT = set(['.','!','?',']',')'])
FOOTER_RE = re.compile(r'^.{5,60} [—–] .+')  # "Post Title — subtitle" cross-reference links


def make_question(title: str) -> str:
    if any(c in title for c in '?.'):
        return f"what do you think about the idea that {title.rstrip('?.')}?"
    return f"what do you think about {title}?"

with open(SRC) as f:
    posts = json.load(f)

examples = []
for post in posts:
    if post["title"].strip() == "Harys Dalvi":
        continue
    title = post["title"].strip()
    body = post["body"]

    chunks = []
    chunk = []
    for paragraph in body.split('\n\n'):
        if not paragraph.strip():
            continue
        if not paragraph.strip()[-1] in PUNCT or len(paragraph.strip().split(' ')) <= 2:
            continue
        if paragraph.strip() == title or paragraph.strip() == "Harys Dalvi":
            continue
        # filter footer lines within the paragraph (cross-reference links)
        lines = [l for l in paragraph.splitlines() if not FOOTER_RE.match(l.strip())]
        paragraph = '\n'.join(lines).strip()
        if not paragraph:
            continue
        chunk.append(paragraph)
        if len(chunk) >= PARAGRAPHS_PER_CHUNK:
            chunks.append('\n\n'.join(chunk).strip())
            chunk = []
    if chunk:
        chunks.append('\n\n'.join(chunk).strip())


    for chunk in chunks:
        text = chunk.strip()
        if not text:
            continue
        examples.append({
            "messages": [
                {"role": "user", "content": make_question(title)},
                {"role": "assistant", "content": text},
            ]
        })

with open(OUT, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Added {len(examples)} blog examples to {OUT}")

