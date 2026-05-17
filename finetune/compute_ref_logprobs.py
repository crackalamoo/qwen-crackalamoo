"""
Pre-compute reference model logprobs for DPO training.

Loads the SFT checkpoint, runs each (prompt, chosen, rejected) pair through it,
and saves per-sequence logprobs back into the JSONL as constants.

Usage:
    ADAPTER=finetune/adapters.npz uv run python finetune/compute_ref_logprobs.py
"""
import json
import os
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

try:
    from finetune.train import inject_lora
except ModuleNotFoundError:
    from train import inject_lora

MODEL_ID = "mlx-community/Qwen3-8B-4bit"
IN = Path(__file__).parent.parent / "dpo_raw.jsonl"
OUT = Path(__file__).parent.parent / "dpo_with_ref.jsonl"


def sequence_logprob(model, tokenizer, prompt_msgs: list, response_text: str) -> tuple[float, list[int], int]:
    """
    Returns (logprob, full_ids, response_start).
    logprob is the sum of log probs over response tokens only.
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = prompt_text + response_text + tokenizer.eos_token

    prompt_ids = tokenizer.encode(prompt_text)
    full_ids = tokenizer.encode(full_text)
    response_start = len(prompt_ids)

    input_ids = mx.array(full_ids)[None]  # [1, T]
    logits = model(input_ids)  # [1, T, vocab_size]
    mx.eval(logits)

    # logits[i] predicts token[i+1], so completion logprobs span [response_start-1, len-1)
    log_probs = mx.log(mx.softmax(logits[0], axis=-1))  # [T, vocab_size]

    total = 0.0
    for i in range(response_start, len(full_ids)):
        token_id = full_ids[i]
        total += log_probs[i - 1, token_id].item()

    return total, full_ids, response_start


def main():
    model, tokenizer = load(MODEL_ID)

    adapter_path = os.environ.get("ADAPTER")
    if adapter_path:
        inject_lora(model, rank=8, alpha=16)
        weights = list(mx.load(adapter_path).items())
        model.load_weights(weights, strict=False)
        mx.eval(model.parameters())
        print(f"Loaded adapter: {adapter_path}")
    else:
        print("No ADAPTER set — using base model as reference")

    with open(IN) as f:
        pairs = [json.loads(l) for l in f]

    print(f"Computing reference logprobs for {len(pairs)} pairs...")

    with open(OUT, "w") as f:
        for i, pair in enumerate(pairs):
            prompt = pair["prompt"]
            chosen_text = pair["chosen"][0]["content"]
            rejected_text = pair["rejected"][0]["content"]

            chosen_logp, chosen_ids, chosen_start = sequence_logprob(model, tokenizer, prompt, chosen_text)
            rejected_logp, rejected_ids, rejected_start = sequence_logprob(model, tokenizer, prompt, rejected_text)

            print(f"[{i+1}/{len(pairs)}] {prompt[0]['content'][:50]}")
            print(f"  ref_chosen_logp:   {chosen_logp:.3f}")
            print(f"  ref_rejected_logp: {rejected_logp:.3f}")
            print(f"  gap: {chosen_logp - rejected_logp:.3f} (negative = model prefers rejected, expected)")

            out = {
                "prompt": prompt,
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
                "chosen_ids": chosen_ids,
                "chosen_response_start": chosen_start,
                "rejected_ids": rejected_ids,
                "rejected_response_start": rejected_start,
                "ref_chosen_logp": chosen_logp,
                "ref_rejected_logp": rejected_logp,
            }
            f.write(json.dumps(out) + "\n")

    print(f"\nWrote {len(pairs)} pairs to {OUT}")


if __name__ == "__main__":
    main()
