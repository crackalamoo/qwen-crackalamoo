"""
Runtime monkeypatches for mlx_lm internals that we depend on but can't fix
upstream without a PR landing + a version bump.
"""

import mlx.core as mx
from mlx.utils import tree_map
import mlx_lm.models.base as _base


def _patched_quantized_sdpa(
    queries: mx.array,
    q_keys: tuple,
    q_values: tuple,
    scale: float,
    mask,
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries = queries * scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        elif n_repeats > 1 and mask.ndim == scores.ndim - 1:
            # mask is missing the head-group axis we inserted into
            # queries/q_keys/q_values above -- add it back so it
            # broadcasts against `scores` (B, n_kv_heads, n_repeats, L, S).
            mask = mx.expand_dims(mask, axis=-3)
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


_base.quantized_scaled_dot_product_attention = _patched_quantized_sdpa
