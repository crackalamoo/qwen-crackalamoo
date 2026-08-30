"""
Continuous batching scheduler.

Multiple requests share a single generation loop. Each request is prefilled
individually, then its KV cache is merged into a shared BatchQuantizedKVCache.
Every scheduler tick runs one batched forward pass over all active sequences.

Memory budget: Qwen3-8B-4bit ~4GB weights + ~144KB/token KV in bf16, roughly
halved by the 8-bit KV cache. At MAX_BATCH=8, max_tokens=512 → ~590MB KV
headroom (bf16 estimate) on 16GB.
"""

import queue
import threading
import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache

from inference import model, tokenizer, sample

from batch_quantized_cache import BatchQuantizedKVCache, KV_GROUP_SIZE, KV_BITS
from prefix_cache import cache, slice_kv_cache

MAX_BATCH = 8
EOS_TOKEN_ID = tokenizer.eos_token_id

# Max tokens processed in one forward pass during prefill
PREFILL_CHUNK_SIZE = 2048

# ~80.6KB/token measured effective KV cost. The 8-bit payload is 72KB/token
# (bf16 is 144KB); the rest is the per-group scales and biases mx.quantize stores
# alongside it.
KV_BYTES_PER_TOKEN = 82 * 1024

# inference.py's mx.set_memory_limit(8GB) is advisory; the real ceiling is Metal's
# ~10.7GB working set. ~4GB weights + ~2.6GB prefix cache (MAX_CACHE_TOKENS at
# the same per-token cost) leaves ~4.1GB for active-batch KV; budget 1.5GB.
ACTIVE_KV_BUDGET_BYTES = int(1.5 * 1024 ** 3)

# "high" admits ahead of "low" when both fit the memory budget, but effective
# priority ages with wait time: after 10s a queued "low" outranks any "high"
# arriving from then on, so it's only overtaken by a bounded set (already-pending
# "high" plus arrivals in its first 10s), never starved indefinitely.
BASE_PRIORITY = {"high": 10.0, "low": 0.0}
AGING_RATE = 1.0  # effective-priority points gained per second of waiting

# Requests over this are rejected before submission
HARD_MAX_PROMPT_TOKENS = 40_960


class PromptTooLargeError(Exception):
    """Raised when a request's tokenized prompt exceeds its effective limit."""

    def __init__(self, prompt_tokens: int, limit: int):
        self.prompt_tokens = prompt_tokens
        self.limit = limit
        super().__init__(f"prompt has {prompt_tokens} tokens, limit is {limit}")

# Both <think> and </think> are single tokens in the Qwen3 vocab
THINK_TOKEN_ID = 151667
THINK_END_TOKEN_ID = 151668
# "</think>\n\n" — what we inject via forced_tokens to force-close thinking
THINK_END_TOKENS = [THINK_END_TOKEN_ID, 271]


@dataclass
class Sequence:
    input_ids: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float = 1.0
    # scheduler writes token IDs here; None signals completion
    token_queue: "queue.Queue[int | None]" = field(default_factory=queue.Queue)
    generated_ids: list[int] = field(default_factory=list)
    # set by _prefill; merged into batch_cache by _add_to_batch
    cache: "list[QuantizedKVCache]" = field(default_factory=list)
    last_token: int = -1
    n_cached_tokens: int = 0  # prompt tokens served from prefix cache
    # queued token IDs to emit instead of sampling (e.g. forced "</think>" closer)
    forced_tokens: list[int] = field(default_factory=list)
    # hard cap on tokens generated inside <think>...</think>; None = unlimited
    thinking_budget: int | None = None
    think_tokens: int = 0  # tokens generated since <think> was seen
    in_think: bool = False  # currently inside a <think>...</think> block
    priority: str = "high"  # "high" (interactive) or "low" (background)
    enqueue_time: float = 0.0  # set by PendingQueue.put(); time.monotonic()


def effective_priority(seq: "Sequence", now: float) -> float:
    """Higher admits first."""
    base = BASE_PRIORITY.get(seq.priority, BASE_PRIORITY["high"])
    wait_s = max(0.0, now - seq.enqueue_time)
    return base + wait_s * AGING_RATE


def should_cache_insert(seq: "Sequence", generated_text: str = "") -> bool:
    """
    Gate for cache.insert(): should this sequence's KV be written into the
    shared radix cache for other requests to reuse? Does not gate cache read.
    """
    if "</think>" in generated_text:
        return False
    if seq.priority == "low":
        return False
    return True


def kv_cost_bytes(seq: "Sequence") -> int:
    """Worst-case KV footprint if seq's cache grows to its max_tokens cap."""
    return (len(seq.input_ids) + seq.max_tokens) * KV_BYTES_PER_TOKEN


class PendingQueue:
    """Thread-safe holding area for not-yet-prefilled sequences.

    Priority-ordered and budget-filtered rather than plain FIFO, so a
    candidate that doesn't fit memory can be skipped in favor of a smaller
    one behind it.
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._items: list[Sequence] = []

    def put(self, seq: Sequence) -> None:
        with self._cv:
            seq.enqueue_time = time.monotonic()
            self._items.append(seq)
            self._cv.notify_all()

    def __len__(self) -> int:
        with self._cv:
            return len(self._items)

    def try_pop_best(self, fits) -> "Sequence | None":
        """Pop the highest-priority sequence matching fits(seq), or None if none fit."""
        with self._cv:
            now = time.monotonic()
            best_idx = None
            best_score = None
            for i, seq in enumerate(self._items):
                if not fits(seq):
                    continue
                score = effective_priority(seq, now)
                if best_score is None or score > best_score:
                    best_idx, best_score = i, score
            if best_idx is None:
                return None
            return self._items.pop(best_idx)

    def pop_best_blocking(self) -> Sequence:
        """Blocks until non-empty, then pops the highest-priority sequence,
        ignoring the memory budget -- used only when active is empty, since
        otherwise an oversized sequence could wait forever."""
        with self._cv:
            while not self._items:
                self._cv.wait()
            now = time.monotonic()
            best_idx = max(range(len(self._items)), key=lambda i: effective_priority(self._items[i], now))
            return self._items.pop(best_idx)


def track_thinking(seq: "Sequence", next_token: int) -> None:
    """
    Update thinking token counter based on the next_token
    and enforce seq.thinking_budget by setting seq.forced_tokens
    once the thinking token budget is reached
    """
    if next_token == THINK_TOKEN_ID:
        seq.in_think = True
    elif next_token == THINK_END_TOKEN_ID:
        seq.in_think = False
    elif seq.in_think:
        seq.think_tokens += 1
        if seq.thinking_budget is not None and seq.think_tokens >= seq.thinking_budget and not seq.forced_tokens:
            # force close thinking on the next tick
            seq.forced_tokens = list(THINK_END_TOKENS)


def prefill_last_token_logits(suffix: list[int], layer_cache: "list[KVCache] | list[QuantizedKVCache]") -> mx.array:
    """
    Runs the prompt suffix through the model in chunks of at most
    PREFILL_CHUNK_SIZE tokens, updating layer_cache in place, and returns
    logits for ONLY the final token position: shape [vocab_size].
    """
    for i in range(0, len(suffix), PREFILL_CHUNK_SIZE):
        chunk = suffix[i : min(len(suffix), i+PREFILL_CHUNK_SIZE)]
        hidden = model.model(mx.array(chunk)[None], cache=layer_cache)
        mx.eval(hidden)

    # hidden: [1, L, 4096] (L = len of the last chunk)
    last_hidden = hidden[:, -1:, :]
    last_logits = model.lm_head(last_hidden) # [1, 1, 151936]
    last_logits = last_logits[0, 0, :]
    return last_logits


def apply_repetition_penalty(logits: mx.array, generated_ids: list[int], penalty: float) -> mx.array:
    """
    Divide logits by `penalty` for every token that has already appeared.
    logits: 1-D array of shape [vocab_size]
    generated_ids: token IDs produced so far for this sequence
    penalty: > 1.0 discourages repetition (1.0 = no-op)
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    unique_ids = mx.array(list(set(generated_ids)))
    mask = (mx.arange(logits.shape[0])[:, None] == unique_ids[None, :]).any(axis=1)
    penalty_vec = mx.where(mask, penalty, 1.0)

    logits = logits / penalty_vec
    return logits


class BatchScheduler:
    def __init__(self):
        self.pending = PendingQueue()
        self.active: list[Sequence] = []
        self._batch_cache = None  # list[BatchQuantizedKVCache] | None
        self._tick = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, seq: Sequence) -> None:
        self.pending.put(seq)

    def _active_kv_bytes(self) -> int:
        return sum(kv_cost_bytes(seq) for seq in self.active)

    def _fits_budget(self, candidate: Sequence) -> bool:
        return self._active_kv_bytes() + kv_cost_bytes(candidate) <= ACTIVE_KV_BUDGET_BYTES

    def _prefill(self, seq: Sequence) -> bool:
        """
        Full prompt forward pass for one sequence.
        Populates seq.cache (per-layer QuantizedKVCache) and samples the first
        generated token. Returns False if the sequence is already done (EOS or
        max_tokens).
        """
        t0 = time.perf_counter()
        kv_from_cache, n_cached_tokens = cache.lookup(seq.input_ids)
        t1 = time.perf_counter()
        if kv_from_cache and n_cached_tokens >= len(seq.input_ids):
            seq.cache = slice_kv_cache(kv_from_cache, len(seq.input_ids) - 1)
            suffix = [seq.input_ids[-1]]
        elif kv_from_cache and n_cached_tokens < len(seq.input_ids):
            seq.cache = kv_from_cache
            suffix = seq.input_ids[n_cached_tokens:]
        else:
            seq.cache = make_prompt_cache(model)
            suffix = seq.input_ids
        seq.n_cached_tokens = n_cached_tokens
        t2 = time.perf_counter()
        last_logits = prefill_last_token_logits(suffix, seq.cache)
        if not isinstance(seq.cache[0], QuantizedKVCache):
            seq.cache = [
                layer.to_quantized(group_size=KV_GROUP_SIZE, bits=KV_BITS)
                for layer in seq.cache
            ]
        mx.eval(last_logits)
        t3 = time.perf_counter()
        if should_cache_insert(seq):
            cache.insert(seq.input_ids, seq.cache)
        t4 = time.perf_counter()
        first_token = int(sample(last_logits, seq.temperature, seq.top_p).item())
        track_thinking(seq, first_token)
        seq.last_token = first_token
        seq.generated_ids.append(first_token)
        seq.token_queue.put(first_token)
        t5 = time.perf_counter()
        print(
            f"[prefill] prompt_len={len(seq.input_ids)} n_cached={n_cached_tokens} "
            f"suffix_len={len(suffix)} | lookup={t1 - t0:.3f}s slice={t2 - t1:.3f}s "
            f"forward+eval={t3 - t2:.3f}s insert={t4 - t3:.3f}s sample={t5 - t4:.3f}s "
            f"total={t5 - t0:.3f}s",
            flush=True,
        )
        if first_token == EOS_TOKEN_ID or len(seq.generated_ids) >= seq.max_tokens:
            seq.token_queue.put(None)
            return False
        return True

    def _add_to_batch(self, seq: Sequence) -> None:
        """
        Merge a prefilled sequence's per-layer QuantizedKVCache into the running batch.

        _batch_cache is list[BatchQuantizedKVCache] — one BatchQuantizedKVCache per
        model layer. Each BatchQuantizedKVCache holds the KV entries for ALL active
        sequences at that layer, left-padded so every sequence's last token is
        right-aligned.
        """
        if self._batch_cache is None:
            # wrap each layer's single QuantizedKVCache into a batch of size 1
            self._batch_cache = [BatchQuantizedKVCache.merge([layer]) for layer in seq.cache]
        else:
            # extend each layer's BatchQuantizedKVCache with the new sequence
            for batch_layer, seq_layer in zip(self._batch_cache, seq.cache):
                batch_layer.extend(BatchQuantizedKVCache.merge([seq_layer]))
        self.active.append(seq)

    def _generation_step(self) -> list[bool]:
        """
        One batched generation step across all active sequences.

        Steps:
          1. Stack each sequence's last_token into a [B, 1] array.
          2. Run model(tokens, cache=self._batch_cache) → logits [B, 1, vocab].
          3. For each sequence i:
               - sample from logits[i, 0, :]
               - update seq.last_token and seq.generated_ids
               - put the token id into seq.token_queue
               - decide if the sequence is done (EOS or max_tokens reached)
               - if done, put None into seq.token_queue
          4. Return a list of bools (True = still alive, False = finished).
        """
        tokens = mx.array([[int(a.last_token)] for a in self.active])

        # model call also updates cache; `logits` itself is unused in the forced_tokens case
        logits = model(tokens, cache=self._batch_cache)
        mx.eval(logits)

        still_alive = []
        for i, seq in enumerate(self.active):
            # if seq.forced_tokens is non-empty, pop and use the front
            # token instead of sampling (still goes through the
            # path below, but bypasses logits sampling entirely).
            if seq.forced_tokens:
                next_token = seq.forced_tokens[0]
                seq.forced_tokens = seq.forced_tokens[1:]
            else:
                token_logits = apply_repetition_penalty(logits[i, 0, :], seq.generated_ids, seq.repetition_penalty)
                next_token = int(sample(token_logits, seq.temperature, seq.top_p).item())

            track_thinking(seq, next_token)

            self.active[i].last_token = next_token
            seq.token_queue.put(next_token)
            seq.generated_ids.append(next_token)
            if len(seq.generated_ids) >= seq.max_tokens or next_token == EOS_TOKEN_ID:
                seq.token_queue.put(None)
                still_alive.append(False)
            else:
                still_alive.append(True)


        return still_alive

    def _run(self):
        """
        Main scheduler loop. Runs in a background thread.
        """
        while True:
            # populate batch: admit by priority, deferring (not rejecting) anything
            # over budget until something active finishes and frees memory.
            while len(self.active) < MAX_BATCH:
                seq = self.pending.try_pop_best(self._fits_budget)
                if seq is None:
                    break # nothing pending fits right now
                seq_alive = self._prefill(seq)
                if seq_alive:
                    self._add_to_batch(seq)

            if not self.active:
                # nothing active to free memory -- admit unconditionally or block forever
                seq = self.pending.pop_best_blocking()
                seq_alive = self._prefill(seq)
                if seq_alive:
                    self._add_to_batch(seq)
                continue # go back to the draw loop

            # batched generation step
            alive_flags = self._generation_step()

            # store finished sequences in prefix cache before removing them
            finished = [i for i, alive in enumerate(alive_flags) if not alive]
            if self._batch_cache is not None:
                for i in finished:
                    seq = self.active[i]
                    generated_text = tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
                    if not should_cache_insert(seq, generated_text):
                        continue
                    full_tokens = seq.input_ids + seq.generated_ids
                    extracted_kv_cache = [layer.extract(i) for layer in self._batch_cache]
                    cache.insert(full_tokens, extracted_kv_cache)


            # remove finished sequences
            keep = [i for i, alive in enumerate(alive_flags) if alive]
            self.active = [self.active[i] for i in keep]
            if keep:
                # partial filter()
                for layer in self._batch_cache:
                    layer.filter(keep)
            else:
                self._batch_cache = None
                mx.clear_cache()  # batch drained — release pool before going idle

            # Cheap safety net: a partial filter() (one sequence in a batch
            # finishing early while others continue) can dump freed
            # buffers into the cache. Clear them here
            self._tick += 1
            if self._tick % 100 == 0 and keep:
                mx.clear_cache()


# module-level singleton — imported by server.py
scheduler = BatchScheduler()


def submit_request(
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    tools: list | None = None,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
    priority: str = "high",
    max_prompt_tokens: int | None = None,
) -> Sequence:
    """Build a Sequence from a chat messages list and submit it to the scheduler."""
    from inference import build_prompt
    input_ids: list[int] = build_prompt(messages, tools=tools, enable_thinking=enable_thinking).tolist()
    effective_limit = HARD_MAX_PROMPT_TOKENS
    if max_prompt_tokens is not None:
        effective_limit = min(max_prompt_tokens, HARD_MAX_PROMPT_TOKENS)
    if len(input_ids) > effective_limit:
        raise PromptTooLargeError(len(input_ids), effective_limit)
    seq = Sequence(
        input_ids=input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        thinking_budget=thinking_budget,
        priority=priority if priority in BASE_PRIORITY else "high",
    )
    scheduler.submit(seq)
    return seq

