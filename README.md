# Crackalamoo Qwen3 Fine-Tuning and Inference

From-scratch LLM fine-tuning and inference stack for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on Apple Silicon, built on [MLX](https://github.com/ml-explore/mlx). Trained to sound like me by training on my blog posts, Discord messages, and hand-written DPO data.

## Components

### Inference (`inference.py`)
Manual token-by-token generation loop with KV cache prefill, top-p sampling, and repetition penalty.

### Batched inference (`batched_inference.py`)
Continuous batching scheduler. Multiple requests share a single generation loop. Each request is prefilled individually, its KV Cache is merged into a shared `BatchKVCache`, and every scheduler tick runs one batched forward pass over all active sequences. Finished sequences are evicted and their cache rows filtered out.

### Serving (`server.py`)
FastAPI server with an OpenAI-compatible `/v1/chat/completions` endpoint. Responses stream via SSE. Includes a minimal chat UI at `/chat`.

### Fine-tuning (`finetune/`)

| File | What it does |
|---|---|
| `train.py` | LoRA SFT — injects low-rank adapters (r=8, α=16) into q/k/v/o projections of every attention layer, then trains on next-token prediction |
| `dpo_train.py` | DPO on top of the SFT checkpoint |
| `compute_ref_logprobs.py` | Computes and caches reference log-probs for each (chosen, rejected) pair |
| `data.py` | Token-budget-aware DataLoader. Packs examples into batches without exceeding a max token count |
| `fetch_alpaca.py`, `blog.py` | Data collection scripts |
| `split.py` | Train/val split with a minimum completion-length filter |
| `sample_dpo.py` | Generates DPO candidate pairs by sampling the SFT model |

## Sample outputs

> **is cursor actually worth it**
>
> I'm not sure<br>
> But I think the LLM is worth it<br>
> And I don't feel like paying for Cursor

> **do you think deep learning is actually just vibes or is there real theory**
>
> I mean it's just vibes for most people<br>
> Theoretical physicists and mathematicians have ideas but I don't think they've cracked the real theory yet<br>
> Maybe it's more of a mix than I thought

> **whats the point of learning math if AI can just do it**
>
> Physical meaning? Not sure<br>
> We use math to describe a lot of physical things I guess lol<br>
> Like the hydrogen atom is an eigenstate with an expectation value?

> **do you think you could get into a top phd program**
>
> You would have a pretty good chance<br>
> I think I probably do but I haven't worked on anything that impressive yet<br>
> Lol

> **whats your read on the whole rust vs c++ debate**
>
> i've been using rust for actual work now that i'm working on games and other things<br>
> but in some cases its actually causing headaches lol and the sheer overhead like 10x slowdown even though its C++ under hood is kinda annoying

## Setup

Requires Python 3.14+ and uv.

Copy `.env.example` to `.env` and adjust as needed:
```bash
cp .env.example .env
```

## Usage

**Start the server:**
```bash
uv run python server.py
```

Then open `http://localhost:8000/chat` (or your configured `QWEN_PORT`) or hit the API:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}], "stream": true}'
```

**Fine-tune (SFT):**
```bash
uv run python finetune/train.py
```

**Fine-tune (DPO), starting from the SFT adapter:**
```bash
ADAPTER=adapters.npz uv run python finetune/dpo_train.py
```

**Run with a LoRA adapter loaded:**
```bash
ADAPTER=adapters.npz uv run python inference.py
```

