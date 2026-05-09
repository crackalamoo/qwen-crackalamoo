import json
import random
import mlx.core as mx

IGNORE_INDEX = -100


class DataLoader:
    def __init__(self, path: str, tokenizer, batch_size: int, token_budget: int):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.token_budget = token_budget
        self.offsets = []
        self.file = open(path, "r")
        while True:
            offset = self.file.tell()
            line = self.file.readline()
            if not line:
                break
            self.offsets.append(offset)

    def _load(self, offset: int) -> tuple[list[int], list[int]]:
        self.file.seek(offset)
        example = json.loads(self.file.readline())
        text = self.tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        input_ids = self.tokenizer.encode(text)
        # encode only the prompt to find where completion begins
        prompt_text = self.tokenizer.apply_chat_template(
            example["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_len = len(self.tokenizer.encode(prompt_text))
        labels = [IGNORE_INDEX] * prompt_len + input_ids[prompt_len:]
        return input_ids, labels

    def __iter__(self):
        random.shuffle(self.offsets)

        def _make_batch(batch):
            pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

            longest = 0
            for src, _tgt in batch:
                longest = max(longest, len(src))

            for j in range(len(batch)):
                src, tgt = batch[j]
                padding = longest - len(src)
                if padding > 0:
                    batch[j][0] = mx.concatenate([mx.array(src), mx.array([pad_id] * padding)])
                    batch[j][1] = mx.concatenate([mx.array(tgt), mx.array([IGNORE_INDEX] * padding)])
                else:
                    batch[j][0] = mx.array(src)
                    batch[j][1] = mx.array(tgt)

            batch = mx.array([b[0] for b in batch]), mx.array([b[1] for b in batch])
            return batch

        for i in range(0, len(self.offsets), self.batch_size):
            window = self.offsets[i:i+self.batch_size]
            examples = [self._load(offset) for offset in window]
            examples.sort(key=lambda e: len(e[0]))


            batch = []
            batch_tokens = 0
            for src, tgt in examples:
                if len(src) > self.token_budget:
                    continue
                elif batch_tokens + len(src) > self.token_budget:
                    if batch:
                        yield _make_batch(batch)
                    batch = []
                    batch_tokens = 0

                batch_tokens += len(src)
                batch.append([src, tgt])


            if batch:
                yield _make_batch(batch)

