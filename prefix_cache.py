"""
Radix tree prefix cache for KV state reuse across requests.

Nodes store a token sequence (edge label), an optional KV snapshot,
and children keyed by first token of each child's edge.

On a cache hit, the caller receives the deepest matching KV snapshot
and the number of tokens it covers, then prefills only the remaining suffix.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models.cache import QuantizedKVCache


MAX_CACHE_TOKENS = 32768  # ~2.0GB quantized KV headroom


def _copy_quantized_layer(layer: QuantizedKVCache, n_tokens: int) -> QuantizedKVCache:
    """Copy one QuantizedKVCache layer, truncated to its first n_tokens."""
    new_layer = QuantizedKVCache(group_size=layer.group_size, bits=layer.bits)
    if layer.keys is not None:
        new_layer.keys, new_layer.values = tree_map(
            lambda x: mx.array(x[..., :n_tokens, :]), (layer.keys, layer.values)
        )
    new_layer.offset = n_tokens
    return new_layer


def _eval_kv_cache(cache: list[QuantizedKVCache]) -> None:
    mx.eval(*[a for l in cache if l.keys is not None for a in (*l.keys, *l.values)])


def _copy_kv_cache(cache: list[QuantizedKVCache]) -> list[QuantizedKVCache]:
    """Deep-copy a list of QuantizedKVCache objects so the caller can mutate without corrupting the stored snapshot."""
    copies = [_copy_quantized_layer(layer, layer.offset) for layer in cache]
    _eval_kv_cache(copies)
    return copies


def slice_kv_cache(cache: list[QuantizedKVCache], n_tokens: int) -> list[QuantizedKVCache]:
    """Return a copy of cache truncated to the first n_tokens."""
    copies = [_copy_quantized_layer(layer, n_tokens) for layer in cache]
    _eval_kv_cache(copies)
    return copies


@dataclass
class RadixNode:
    tokens: list[int]
    kv_cache: Optional[list[QuantizedKVCache]] = None
    children: dict[int, "RadixNode"] = field(default_factory=dict)
    last_access: float = field(default_factory=time.monotonic)

    def token_count(self) -> int:
        """Total tokens covered by kv_cache at this node (0 if no cache)."""
        if self.kv_cache is None:
            return 0
        return self.kv_cache[0].offset if self.kv_cache else 0


class RadixCache:
    """
    Radix tree mapping token prefix sequences to KV cache snapshots.
    Thread-safe via a single lock (the scheduler calls this from its thread).
    """

    def __init__(self):
        self._root = RadixNode(tokens=[])
        self._lock = threading.Lock()
        self._total_cached_tokens = 0

    def lookup(self, tokens: list[int]) -> tuple[Optional[list[QuantizedKVCache]], int]:
        """
        Find the longest cached prefix of `tokens`.

        Returns (kv_cache_copy, match_depth) where:
          - kv_cache_copy is a fresh copy of the best matching KV snapshot (or None)
          - match_depth is how many tokens that snapshot covers
        """
        with self._lock:
            best_cache = None
            depth = 0
            best_depth = 0
            node: RadixNode = self._root
            while node.children:
                child = node.children.get(tokens[depth], None)
                if child is None:
                    break

                shared = 1
                while shared < min(len(child.tokens), len(tokens) - depth) and tokens[depth+shared] == child.tokens[shared]:
                    shared += 1

                child.last_access = time.time()
                if shared == len(child.tokens):
                    depth += len(child.tokens)
                    if child.kv_cache:
                        best_cache = _copy_kv_cache(child.kv_cache)
                        best_depth = depth
                    node = child
                else:
                    depth += shared
                    if child.kv_cache:
                        best_depth = depth
                        best_cache = slice_kv_cache(child.kv_cache, best_depth)
                    break

                if depth >= len(tokens):
                    break
            return (best_cache, best_depth)

    def insert(self, tokens: list[int], kv_cache: list[QuantizedKVCache]) -> None:
        """
        Insert a (tokens, kv_cache) entry into the tree.
        """
        if kv_cache and kv_cache[0].offset > MAX_CACHE_TOKENS:
            return
        with self._lock:
            _eval_kv_cache(kv_cache)

            def _insert_into(node, depth=0):
                child = node.children.get(tokens[depth], None)
                if child is None:
                    # insert new node
                    node.children[tokens[depth]] = RadixNode(
                        tokens[depth:],
                        kv_cache,
                        dict(),
                        time.time()
                    )
                    self._total_cached_tokens += kv_cache[0].offset
                    self._maybe_evict()
                    return

                if child.tokens == tokens[depth:]:
                    # node already exists exactly
                    child.last_access = time.time()
                    return

                if depth+len(child.tokens) > len(tokens) or child.tokens != tokens[depth:depth+len(child.tokens)]:
                    # partial match, need to split
                    shared_prefix = []
                    i = 0
                    while i < min(len(tokens) - depth, len(child.tokens)) and tokens[depth+i] == child.tokens[i]:
                        i += 1
                    shared_prefix = child.tokens[:i]
                    node_1 = tokens[depth+i:]
                    node_2 = child.tokens[i:]

                    child_1 = RadixNode(
                        node_1,
                        kv_cache,
                        dict(),
                        time.time()
                    )
                    child_2 = RadixNode(
                        node_2,
                        child.kv_cache,
                        child.children,
                        child.last_access  # preserve old node's access time so LRU evicts stale branch first
                    )
                    children = {
                            node_1[0]: child_1,
                            node_2[0]: child_2
                    } if node_1 else {
                        node_2[0]: child_2
                    }
                    node.children[tokens[depth]] = RadixNode(
                        shared_prefix,
                        None if node_1 else kv_cache,
                        children,
                        time.time()
                    )
                    self._total_cached_tokens += kv_cache[0].offset
                    self._maybe_evict()
                    return

                # if child is a full prefix match, keep traversing
                if child.tokens == tokens[depth:depth+len(child.tokens)]:
                    return _insert_into(child, depth + len(child.tokens))

            _insert_into(self._root)

        # Each finished generation frees its BatchKVCache into
        # MLX's cache pool, where it sits useless since the next request's KV
        # shapes differ. Clear cache outside the lock so the next
        # lookup() calls can start without waiting for cache clear.
        mx.clear_cache()

    def _maybe_evict(self) -> None:
        """Evict LRU leaves until _total_cached_tokens <= MAX_CACHE_TOKENS."""
        while self._total_cached_tokens > MAX_CACHE_TOKENS:
            leaf, leaf_path = self._find_lru_leaf(self._root)
            if leaf is None:
                break
            self._evict_leaf(leaf, leaf_path)


    def _find_lru_leaf(self, node: RadixNode, path: Optional[list[RadixNode]]=None) -> Optional["RadixNode"]:
        """Return the node with the oldest last_access and kv_cache != None."""
        path = (path or []) + [node]
        best = node if node.kv_cache is not None else None
        best_path = path if best is not None else None
        for child in node.children.values():
            candidate, candidate_path = self._find_lru_leaf(child, path)
            if candidate is not None:
                if best is None or candidate.last_access < best.last_access:
                    best = candidate
                    best_path = candidate_path
        return best, best_path

    def _evict_leaf(self, leaf: RadixNode, leaf_path: list[RadixNode]) -> None:
        """Remove a node's KV cache and update token count."""
        if leaf.kv_cache is not None:
            self._total_cached_tokens -= leaf.token_count()
            leaf.kv_cache = None
            if leaf_path:
                for i in reversed(range(1, len(leaf_path))):
                    parent = leaf_path[i-1]
                    child = leaf_path[i]
                    if not child.kv_cache and not child.children:
                        del parent.children[child.tokens[0]]


# module-level singleton
cache = RadixCache()

