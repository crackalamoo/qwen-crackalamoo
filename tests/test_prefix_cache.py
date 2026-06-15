"""Tests for the radix-tree prefix cache (prefix_cache.py)."""

import pytest
from mlx_lm.models.cache import QuantizedKVCache

from prefix_cache import RadixCache


def fake_kv(n_tokens: int) -> list[QuantizedKVCache]:
    """A minimal stand-in for a real KV cache snapshot covering n_tokens tokens."""
    layer = QuantizedKVCache()
    layer.offset = n_tokens
    return [layer]


@pytest.fixture
def cache():
    return RadixCache()


def test_lookup_empty_tree_returns_no_match(cache):
    kv, depth = cache.lookup([1, 2, 3])
    assert kv is None
    assert depth == 0


def test_insert_then_exact_lookup(cache):
    tokens = [1, 2, 3, 4]
    cache.insert(tokens, fake_kv(4))

    kv, depth = cache.lookup(tokens)
    assert depth == 4
    assert kv is not None
    assert kv[0].offset == 4


def test_lookup_returns_prefix_match_for_longer_query(cache):
    cache.insert([1, 2, 3, 4], fake_kv(4))

    kv, depth = cache.lookup([1, 2, 3, 4, 5, 6])
    assert depth == 4
    assert kv[0].offset == 4


def test_lookup_no_shared_first_token(cache):
    cache.insert([1, 2, 3, 4], fake_kv(4))

    kv, depth = cache.lookup([9, 9, 9])
    assert kv is None
    assert depth == 0


def test_insert_split_on_diverging_prefix(cache):
    cache.insert([1, 2, 3, 4], fake_kv(4))
    cache.insert([1, 2, 5, 6], fake_kv(4))

    kv1, depth1 = cache.lookup([1, 2, 3, 4])
    assert depth1 == 4
    kv2, depth2 = cache.lookup([1, 2, 5, 6])
    assert depth2 == 4
    kv3, depth3 = cache.lookup([1, 2, 3, 9])
    assert kv3[0].offset == 3
    assert depth3 == 3
