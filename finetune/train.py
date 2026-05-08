import mlx.core as mx
import mlx.nn as nn
import time
from pathlib import Path
from mlx.optimizers import AdamW
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx.utils import tree_flatten

try:
    from finetune.data import DataLoader
except ModuleNotFoundError:
    from data import DataLoader

MODEL_ID = "mlx-community/Qwen3-8B-4bit"
LORA_RANK = 8
LORA_ALPHA = 16
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
EPOCHS = 2
TRAIN_PATH = str(Path(__file__).parent.parent / "train.jsonl")
VAL_PATH = str(Path(__file__).parent.parent / "val.jsonl")
VAL_EVERY = 250
ADAPTER_PATH = str(Path(__file__).parent / "adapters.npz")


def inject_lora(model, rank: int, alpha: int):
    # replace q/k/v/o projections in every attention layer with LoRALinear
    for layer in model.model.layers:
        attn = layer.self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            linear = getattr(attn, name)
            setattr(attn, name, LoRALinear.from_base(linear, r=rank, scale=alpha / rank))


def loss_fn(model, input_ids: mx.array, labels: mx.array) -> mx.array:
    # input_ids: [B, T], labels: [B, T] with -100 for masked positions
    # labels are shifted: position i predicts position i+1
    logits = model(input_ids)  # [B, T, vocab_size]

    # logits[j] predicts token j+1, so shift by 1
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    has_label = shift_labels != -100
    targets = mx.where(has_label, shift_labels, 0)
    loss = nn.losses.cross_entropy(shift_logits, targets)
    loss = mx.where(has_label, loss, 0.0)

    return mx.sum(loss) / mx.sum(has_label)


def train(max_steps: int = None):
    model, tokenizer = load(MODEL_ID)
    inject_lora(model, rank=LORA_RANK, alpha=LORA_ALPHA)

    # freeze everything, then unfreeze only lora_a and lora_b
    model.freeze()
    for _name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze(keys=["lora_a", "lora_b"])

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total = sum(p.size for _, p in tree_flatten(model.parameters()))
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = AdamW(learning_rate=LEARNING_RATE)
    loader = DataLoader(TRAIN_PATH, tokenizer, batch_size=BATCH_SIZE)
    val_loader = DataLoader(VAL_PATH, tokenizer, batch_size=BATCH_SIZE)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    t_start = time.time()

    for epoch in range(EPOCHS):
        for step, (input_ids, labels) in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break
            loss, grads = loss_and_grad(model, input_ids, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            if step % VAL_EVERY == 0 and step > 0:
                val_losses = []
                for val_ids, val_labels in val_loader:
                    val_loss = loss_fn(model, val_ids, val_labels)
                    mx.eval(val_loss)
                    val_losses.append(val_loss.item())
                print(f"  val loss: {sum(val_losses)/len(val_losses):.4f}")

    elapsed = time.time() - t_start
    print(f"Training time: {elapsed/3600:.2f}h ({elapsed:.0f}s)")

    # save only the LoRA adapter weights
    lora_weights = {k: v for k, v in tree_flatten(model.trainable_parameters())}
    mx.savez(ADAPTER_PATH, **lora_weights)
    print(f"Saved adapters to {ADAPTER_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    train(max_steps=args.max_steps)

