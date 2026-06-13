"""
BatchQuantizedKVCache -- batched, 8-bit-quantized KV cache for continuous batching.

Scope: Qwen3 only (dense GQA, plain causal attention, no sliding window / no
MLA). DeepSeek-style MLA caches (k_pe) are out of scope.

Dependency: requires a 2-line patch to quantized_scaled_dot_product_attention
in mlx_lm/models/base.py (applied locally to .venv site-packages -- reapply
if the venv is rebuilt) that expands the mask to 5D for GQA. Without it,
batched quantized attention with n_repeats > 1 either errors or silently
misaligns masks across batch rows.
"""

import mlx.core as mx
from mlx_lm.models.base import create_causal_mask
from mlx.utils import tree_map
from mlx_lm.models.cache import QuantizedKVCache, _BaseCache


class BatchQuantizedKVCache(_BaseCache):
    step = 256

    def __init__(self, left_padding: list[int], group_size: int = 64, bits: int = 8):
        self.keys = None    # (packed, scales, biases) tuple, or None
        self.values = None  # (packed, scales, biases) tuple, or None
        self.group_size = group_size
        self.bits = bits
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    @classmethod
    def merge(cls, caches: "list[QuantizedKVCache]") -> "BatchQuantizedKVCache":
        """
        Combine N single-sequence QuantizedKVCache instances (one per
        newly-prefilled request) into one BatchQuantizedKVCache, analogous
        to BatchKVCache.merge but generalized to the 6 (packed, scales,
        biases) x (keys, values) arrays each QuantizedKVCache holds.
        """
        group_size, bits = caches[0].group_size, caches[0].bits
        if any(c.group_size != group_size or c.bits != bits for c in caches):
            raise ValueError(
                "Cannot merge QuantizedKVCache objects with different "
                "quantization settings"
            )

        lengths = [c.offset for c in caches]
        max_length = max(lengths)

        # No cache has content so make an empty one
        if max_length == 0:
            return BatchQuantizedKVCache([0] * len(caches))

        padding = [max_length - l for l in lengths]
        B = len(caches)
        H = max(c.keys[0].shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys[0].shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values[0].shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys[0].dtype for c in caches if c.keys is not None))
        n_groups_k = max(c.keys[1].shape[3] for c in caches if c.keys is not None)
        n_groups_v = max(c.values[1].shape[3] for c in caches if c.values is not None)
        dt_group = next(iter(c.keys[1].dtype for c in caches if c.keys is not None))

        keys = (
            mx.zeros((B, H, max_length, Dk), dtype=dt),
            mx.zeros((B, H, max_length, n_groups_k), dtype=dt_group),
            mx.zeros((B, H, max_length, n_groups_k), dtype=dt_group),
        )
        values = (
            mx.zeros((B, H, max_length, Dv), dtype=dt),
            mx.zeros((B, H, max_length, n_groups_v), dtype=dt_group),
            mx.zeros((B, H, max_length, n_groups_v), dtype=dt_group),
        )
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            for j, (k, v) in enumerate(zip(c.keys, c.values)):
                keys[j][i : i + 1, :, p : p + c.offset] = k[..., : c.offset, :]
                values[j][i : i + 1, :, p : p + c.offset] = v[..., : c.offset, :]

        cache = cls(padding, group_size, bits)
        cache.keys = keys
        cache.values = values
        cache.offset += keys[0].shape[2]
        cache._idx = keys[0].shape[2]

        return cache

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps
        self._idx += num_steps

        keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self._idx, :] = keys[i]
            self.values[i][..., prev : self._idx, :] = values[i]

        return tree_map(lambda x: x[..., : self._idx, :], (self.keys, self.values))

    def filter(self, batch_indices):
        """
        In-place filter to keep just the given indices in the cache.
        """
        if self.keys is not None:
            self.keys, self.values = tree_map(
                lambda x: x[batch_indices], (self.keys, self.values)
            )
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        # Shift left to reduce padding
        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys, self.values = tree_map(
                    lambda x: x[..., min_left_pad:, :], (self.keys, self.values)
                )
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        """
        In-place extend this cache with the other cache.
        """
        if self.group_size != other.group_size or self.bits != other.bits:
            raise ValueError(
                "Cannot extend BatchQuantizedKVCache with different "
                "quantization settings"
            )

        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys[0].shape
            M = self.values[0].shape[3]
            dt = self.keys[0].dtype
            N_g = self.keys[1].shape[3]
            dt_q = self.keys[1].dtype
        if other.keys is not None:
            B, H, L2, D = other.keys[0].shape
            M = other.values[0].shape[3]
            dt = other.keys[0].dtype
            N_g = other.keys[1].shape[3]
            dt_q = other.keys[1].dtype
        max_size = max(L1, L2)

        # Pad the keys and values so they are right-justified
        # with the index and the same size
        def pad(c):
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = (
                    mx.array([], dtype=dt).reshape(Bc, H, 0, D),
                    mx.array([], dtype=dt_q).reshape(Bc, H, 0, N_g),
                    mx.array([], dtype=dt_q).reshape(Bc, H, 0, N_g),
                )
                v = (
                    mx.array([], dtype=dt).reshape(Bc, H, 0, M),
                    mx.array([], dtype=dt_q).reshape(Bc, H, 0, N_g),
                    mx.array([], dtype=dt_q).reshape(Bc, H, 0, N_g),
                )
            left = max_idx - c._idx
            right = max_size - k[0].shape[2] - left
            if right < 0:
                k, v = tree_map(
                    lambda x: x[..., :right, :], (k, v)
                )
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k, v = tree_map(
                    lambda x: mx.pad(x, pad), (k, v)
                )
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        ps = pad(self)
        po = pad(other)
        self.keys = tree_map(
            lambda a, b: mx.concatenate([a, b], axis=0),
            ps[0], po[0]
        )
        self.values = tree_map(
            lambda a, b: mx.concatenate([a, b], axis=0),
            ps[1], po[1]
        )
        self.offset = mx.concatenate([ps[2], po[2]])
        self.left_padding = mx.concatenate([ps[3], po[3]])
        self._idx = max_idx

    def extract(self, idx):
        cache = QuantizedKVCache(group_size=self.group_size, bits=self.bits)
        padding = self.left_padding[idx].item()
        cache.keys, cache.values = tree_map(
            lambda x: mx.contiguous(x[idx : idx + 1, :, padding : self._idx]), (self.keys, self.values)
        )
        cache.offset = cache.keys[0].shape[2]
        return cache

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return sum(x.nbytes for x in (*self.keys, *self.values))

