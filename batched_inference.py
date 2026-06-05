"""
Continuous batching scheduler.

Multiple requests share a single generation loop. Each request is prefilled
individually, then its KV cache is merged into a shared BatchKVCache. Every
scheduler tick runs one batched forward pass over all active sequences.

Memory budget: Qwen3-8B-4bit ~4GB weights + ~144KB/token KV.
At MAX_BATCH=8, max_tokens=512 → ~590MB KV headroom on 16GB. Don't raise
MAX_BATCH without checking memory first.
"""

import queue
import threading
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache

from inference import model, tokenizer, sample

MAX_BATCH = 8
EOS_TOKEN_ID = tokenizer.eos_token_id


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
    cache: "list[KVCache]" = field(default_factory=list)
    last_token: int = -1


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
        self.pending: queue.Queue[Sequence] = queue.Queue()
        self.active: list[Sequence] = []
        self._batch_cache = None  # BatchKVCache | None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, seq: Sequence) -> None:
        self.pending.put(seq)

    def _prefill(self, seq: Sequence) -> bool:
        """
        Full prompt forward pass for one sequence.
        Populates seq.cache (per-layer KVCache) and samples the first generated
        token. Returns False if the sequence is already done (EOS or max_tokens).
        """
        seq.cache = make_prompt_cache(model)
        logits = model(mx.array(seq.input_ids)[None], cache=seq.cache)
        mx.eval(logits)
        first_token = int(sample(logits[0, -1, :], seq.temperature, seq.top_p).item())
        seq.last_token = first_token
        seq.generated_ids.append(first_token)
        seq.token_queue.put(first_token)
        if first_token == EOS_TOKEN_ID or len(seq.generated_ids) >= seq.max_tokens:
            seq.token_queue.put(None)
            return False
        return True

    def _add_to_batch(self, seq: Sequence) -> None:
        """
        Merge a prefilled sequence's per-layer KVCache into the running batch.

        _batch_cache is list[BatchKVCache] — one BatchKVCache per model layer.
        Each BatchKVCache holds the KV entries for ALL active sequences at that layer,
        left-padded so every sequence's last token is right-aligned.
        """
        if self._batch_cache is None:
            # wrap each layer's single KVCache into a batch of size 1
            self._batch_cache = [KVCache.merge([layer]) for layer in seq.cache]
        else:
            # extend each layer's BatchKVCache with the new sequence
            for batch_layer, seq_layer in zip(self._batch_cache, seq.cache):
                batch_layer.extend(KVCache.merge([seq_layer]))
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

        logits = model(tokens, cache=self._batch_cache)
        mx.eval(logits)

        still_alive = []
        for i, seq in enumerate(self.active):
            token_logits = apply_repetition_penalty(logits[i, 0, :], seq.generated_ids, seq.repetition_penalty)
            next_token = int(sample(token_logits, seq.temperature, seq.top_p).item())
            self.active[i].last_token = next_token
            seq.token_queue.put(next_token)
            seq.generated_ids.append(int(next_token))
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
            # populate batch
            while len(self.active) < MAX_BATCH:
                try:
                    seq = self.pending.get_nowait()
                    seq_alive = self._prefill(seq)
                    if seq_alive:
                        self._add_to_batch(seq)
                except queue.Empty:
                    break # no sequences to work with

            if not self.active:
                seq = self.pending.get() # blocks here
                seq_alive = self._prefill(seq)
                if seq_alive:
                    self._add_to_batch(seq)
                continue # go back to the draw loop

            # batched generation step
            alive_flags = self._generation_step()

            # remove finished sequences
            keep = [i for i, alive in enumerate(alive_flags) if alive]
            self.active = [self.active[i] for i in keep]
            if keep:
                for layer in self._batch_cache:
                    layer.filter(keep)
            else:
                self._batch_cache = None


# module-level singleton — imported by server.py
scheduler = BatchScheduler()


def submit_request(
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    tools: list | None = None,
) -> Sequence:
    """Build a Sequence from a chat messages list and submit it to the scheduler."""
    from inference import build_prompt
    input_ids: list[int] = build_prompt(messages, tools=tools).tolist()
    seq = Sequence(
        input_ids=input_ids,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    scheduler.submit(seq)
    return seq

