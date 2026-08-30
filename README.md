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

## Benchmarks

Qwen3-8B-4bit on an M2 Pro Mac mini (16GB, 200 GB/s). Weights 4.29GB resident, KV cache 80.6 KB/token measured.

| Context | Decode | Bandwidth limit | Prefill | TTFT |
|---|---|---|---|---|
| 60 | 27.2 tok/s | 43.4 tok/s (63%) | — | 0.4s |
| 3.3k | 22.0 tok/s | 41.0 tok/s (54%) | 157 tok/s | 21s |
| 13k | 14.5 tok/s | 35.2 tok/s (41%) | 124 tok/s | 108s |

Decode is memory-bandwidth-bound: every step reads all 4.29GB of weights plus the whole KV
cache onto the chip, so 200 GB/s sets a tokens/second limit.

A 13k-token prompt takes **108s cold, 7.5s on a cache hit**.

Concurrent requests (short prompts, aggregate across the batch):

| Batch | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| tok/s | 26.8 | 34.2 | 38.6 | 34.9 |

Batching amortizes the weight read poorly. Measured in isolation, `mx.quantized_matmul`
costs 2.4x more at batch 4 than at batch 1, but 5.4x more at batch 8, explaining why
amortized throughput actually drops off.

## Sample outputs (fine-tuned)

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

