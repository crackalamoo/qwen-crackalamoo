"""
DPO fine-tuning on top of the SFT LoRA checkpoint.

Usage:
    ADAPTER=finetune/adapters.npz uv run python finetune/dpo_train.py
"""
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear

try:
    from finetune.train import inject_lora
except ModuleNotFoundError:
    from train import inject_lora

import os

MODEL_ID = "mlx-community/Qwen3-8B-4bit"
DATA_PATH = Path(__file__).parent.parent / "dpo_with_ref.jsonl"
OUT_ADAPTER = Path(__file__).parent / "dpo_adapters.npz"

BETA = 0.1
LEARNING_RATE = 5e-6
EPOCHS = 5
SEED = 42
VAL_FRACTION = 0.2  # hold out ~3 pairs for eval


def sequence_logprob(model, input_ids: mx.array, response_start: int) -> mx.array:
    """Sum of log probs over response tokens only."""
    logits = model(input_ids[None])[0]  # [T, vocab]
    # numerically stable log softmax
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # [T, vocab]

    # token at position j is predicted by logits at position j-1
    # response tokens: positions response_start..T-1
    response_ids = input_ids[response_start:]          # [R]
    logit_rows = log_probs[response_start - 1:-1]      # [R, vocab]

    # gather: one-hot select the actual token's log prob at each position
    vocab_size = logit_rows.shape[-1]
    one_hot = (mx.arange(vocab_size) == response_ids[:, None]).astype(mx.float32)  # [R, vocab]
    token_logps = mx.sum(logit_rows * one_hot, axis=-1)  # [R]
    return mx.sum(token_logps)


def dpo_loss(model, batch: dict) -> mx.array:
    chosen_ids = mx.array(batch["chosen_ids"])
    rejected_ids = mx.array(batch["rejected_ids"])
    ref_chosen = batch["ref_chosen_logp"]
    ref_rejected = batch["ref_rejected_logp"]

    pi_chosen = sequence_logprob(model, chosen_ids, batch["chosen_response_start"])
    pi_rejected = sequence_logprob(model, rejected_ids, batch["rejected_response_start"])

    chosen_reward = BETA * (pi_chosen - ref_chosen)
    rejected_reward = BETA * (pi_rejected - ref_rejected)

    loss = -nn.log_sigmoid(chosen_reward - rejected_reward)
    return loss, pi_chosen, pi_rejected


def main():
    model, _tokenizer = load(MODEL_ID)
    inject_lora(model, rank=8, alpha=16)

    adapter_path = os.environ.get("ADAPTER")
    if adapter_path:
        weights = list(mx.load(adapter_path).items())
        model.load_weights(weights, strict=False)
        mx.eval(model.parameters())
        print(f"Initialized policy from: {adapter_path}")

    model.freeze()
    for _name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze(keys=["lora_a", "lora_b"])

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"Trainable params: {trainable:,}")

    with open(DATA_PATH) as f:
        pairs = [json.loads(l) for l in f]

    random.seed(SEED)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRACTION))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"Train: {len(train_pairs)} pairs, Val: {len(val_pairs)} pairs")

    optimizer = AdamW(learning_rate=LEARNING_RATE)
    loss_and_grad = nn.value_and_grad(model, lambda m, b: dpo_loss(m, b)[0])

    t_start = time.time()

    for epoch in range(EPOCHS):
        random.shuffle(train_pairs)
        epoch_losses = []

        for step, batch in enumerate(train_pairs):
            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_val = loss.item()
            epoch_losses.append(loss_val)

            # also compute margin for insight
            _, pi_c, pi_r = dpo_loss(model, batch)
            mx.eval(pi_c, pi_r)
            margin = pi_c.item() - pi_r.item()
            print(f"epoch {epoch} step {step} | loss {loss_val:.4f} | margin {margin:.2f}")

        # val
        val_losses = []
        for batch in val_pairs:
            loss, pi_c, pi_r = dpo_loss(model, batch)
            mx.eval(loss)
            val_losses.append(loss.item())
        avg_val = sum(val_losses) / len(val_losses)
        avg_train = sum(epoch_losses) / len(epoch_losses)
        print(f"=== epoch {epoch} | train_loss {avg_train:.4f} | val_loss {avg_val:.4f} ===")

    elapsed = time.time() - t_start
    print(f"Training time: {elapsed:.1f}s")

    lora_weights = {k: v for k, v in tree_flatten(model.trainable_parameters())}
    mx.savez(str(OUT_ADAPTER), **lora_weights)
    print(f"Saved DPO adapter to {OUT_ADAPTER}")


if __name__ == "__main__":
    main()
