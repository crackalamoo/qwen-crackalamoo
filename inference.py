import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODEL_ID = "mlx-community/Qwen3-8B-4bit"

model, tokenizer = load(MODEL_ID)

# optionally load a LoRA adapter: set ADAPTER env var to path/to/adapters.npz
import os
_adapter_path = os.environ.get("ADAPTER")
if _adapter_path:
    from finetune.train import inject_lora
    inject_lora(model, rank=8, alpha=16)
    weights = list(mx.load(_adapter_path).items())
    model.load_weights(weights, strict=False)
    mx.eval(model.parameters())
    print(f"Loaded adapter: {_adapter_path}")


def build_prompt(messages: list[dict], tools: list | None = None, enable_thinking: bool = False) -> mx.array:
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    if tools is not None:
        kwargs["tools"] = tools
    text = tokenizer.apply_chat_template(messages, **kwargs)
    tokens = tokenizer.encode(text)
    return mx.array(tokens)


def sample(logits: mx.array, temperature: float, top_p: float) -> mx.array:
    probs = mx.softmax(logits / temperature)
    indices = mx.argsort(probs)[::-1]
    sorted_probs = probs[indices]
    cumsum = mx.cumsum(sorted_probs)
    mask = (cumsum - sorted_probs) < top_p
    sorted_probs = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))
    sampled = mx.random.categorical(mx.log(sorted_probs + 1e-9))
    return indices[sampled]


def generate(messages: list[dict], max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, repetition_penalty=1.1):
    input_ids = build_prompt(messages)
    cache = make_prompt_cache(model)

    # prefill
    logits = model(input_ids[None], cache=cache)

    generated_ids = []
    decoded_so_far = ""

    for _ in range(max_tokens):
        next_token_logits = logits[0, -1, :]  # (vocab_size,) — only the last position

        if repetition_penalty != 1.0:
            penalty_vec = mx.ones(next_token_logits.shape[0])
            apply_penalty_array = mx.array(repetition_penalty)
            token_id_range = mx.arange(next_token_logits.shape[0])
            for token_id in set(generated_ids):
                penalty_vec = mx.where(
                    token_id_range == token_id,
                    apply_penalty_array,
                    penalty_vec,
                )
            next_token_logits = next_token_logits / penalty_vec

        next_token = sample(next_token_logits, temperature, top_p)
        mx.eval(next_token)

        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)
        new_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        safe_text = new_text.rstrip('�')
        if len(safe_text) > len(decoded_so_far):
            yield safe_text[len(decoded_so_far):]
            decoded_so_far = safe_text

        # prepare for next iteration
        logits = model(mx.array([[token_id]]), cache=cache)  # (1, seq_len, vocab_size)


if __name__ == "__main__":
    prompt = "Explain what a KV cache is in one paragraph."
    messages = [{"role": "user", "content": prompt}]
    print(f"Prompt: {prompt}\n")
    for chunk in generate(messages):
        print(chunk, end="", flush=True)
    print()
