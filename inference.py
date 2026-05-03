import mlx.core as mx
from mlx_lm import load

MODEL_ID = "mlx-community/Qwen3-8B-4bit"

model, tokenizer = load(MODEL_ID)


def build_prompt(user_message: str) -> mx.array:
    messages = [{"role": "user", "content": user_message}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
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


def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
    input_ids = build_prompt(prompt)
    tokens = input_ids

    for _ in range(max_tokens):
        logits = model(tokens[None])  # (1, seq_len, vocab_size)
        next_token_logits = logits[0, -1, :]  # (vocab_size,) — only the last position

        next_token = sample(next_token_logits, temperature, top_p)
        mx.eval(next_token)

        token_id = next_token.item()

        if token_id == tokenizer.eos_token_id:
            break

        tokens = mx.concatenate([tokens, mx.array([token_id])])
        yield tokenizer.decode([token_id])


if __name__ == "__main__":
    prompt = "Explain what a KV cache is in one paragraph."
    print(f"Prompt: {prompt}\n")
    for chunk in generate(prompt):
        print(chunk, end="", flush=True)
    print()
