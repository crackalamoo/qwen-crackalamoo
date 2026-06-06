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
from mlx_lm.models.cache import KVCache


MAX_CACHE_TOKENS = 16384  # ~2.3GB KV headroom


def _copy_kv_cache(cache: list[KVCache]) -> list[KVCache]:
    """Deep-copy a list of KVCache objects so the caller can mutate without corrupting the stored snapshot."""
    copies = []
    for layer in cache:
        new_layer = KVCache()
        if layer.keys is not None:
            new_layer.keys = mx.array(layer.keys[..., :layer.offset, :])
            new_layer.values = mx.array(layer.values[..., :layer.offset, :])
        new_layer.offset = layer.offset
        copies.append(new_layer)
    mx.eval(*[l.keys for l in copies if l.keys is not None],
            *[l.values for l in copies if l.values is not None])
    return copies


def slice_kv_cache(cache: list[KVCache], n_tokens: int) -> list[KVCache]:
    """Return a copy of cache truncated to the first n_tokens."""
    copies = []
    for layer in cache:
        new_layer = KVCache()
        if layer.keys is not None:
            new_layer.keys = mx.array(layer.keys[..., :n_tokens, :])
            new_layer.values = mx.array(layer.values[..., :n_tokens, :])
        new_layer.offset = n_tokens
        copies.append(new_layer)
    mx.eval(*[l.keys for l in copies if l.keys is not None],
            *[l.values for l in copies if l.values is not None])
    return copies


@dataclass
class RadixNode:
    tokens: list[int]
    kv_cache: Optional[list[KVCache]] = None
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

    def lookup(self, tokens: list[int]) -> tuple[Optional[list[KVCache]], int]:
        """
        Find the longest cached prefix of `tokens`.

        Returns (kv_cache_copy, match_depth) where:
          - kv_cache_copy is a fresh copy of the best matching KV snapshot (or None)
          - match_depth is how many tokens that snapshot covers
        """
        with self._lock:
            best_cache = None
            best_depth = 0
            node: RadixNode = self._root
            while node.children:
                child = node.children.get(tokens[best_depth], None)
                if child is None:
                    break

                shared = 1
                while shared < min(len(child.tokens), len(tokens) - best_depth) and tokens[best_depth+shared] == child.tokens[shared]:
                    shared += 1

                child.last_access = time.time()
                if shared == len(child.tokens):
                    if child.kv_cache:
                        best_cache = _copy_kv_cache(child.kv_cache)
                    best_depth += len(child.tokens)
                    node = child
                else:
                    if child.kv_cache:
                        best_cache = slice_kv_cache(child.kv_cache, best_depth + shared)
                    best_depth += shared
                    break

                if best_depth >= len(tokens):
                    break
            return (best_cache, best_depth)

    def insert(self, tokens: list[int], kv_cache: list[KVCache]) -> None:
        """
        Insert a (tokens, kv_cache) entry into the tree.
        """
        if kv_cache and kv_cache[0].offset > MAX_CACHE_TOKENS:
            return
        with self._lock:
            mx.eval(*[layer.keys for layer in kv_cache if layer.keys is not None],
                    *[layer.values for layer in kv_cache if layer.values is not None])

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
                        time.time()
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

    def _maybe_evict(self) -> None:
        """Evict LRU leaves until _total_cached_tokens <= MAX_CACHE_TOKENS."""
        while self._total_cached_tokens > MAX_CACHE_TOKENS:
            leaf = self._find_lru_leaf(self._root)
            if leaf is None:
                break
            self._evict_leaf(leaf)

    def _find_lru_leaf(self, node: RadixNode) -> Optional["RadixNode"]:
        """Return the leaf node with the oldest last_access time."""
        if not node.children:
            return node if node.kv_cache is not None else None
        candidates = [self._find_lru_leaf(child) for child in node.children.values()]
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.last_access)

    def _evict_leaf(self, leaf: RadixNode) -> None:
        """Remove a leaf node's KV cache and update token count."""
        if leaf.kv_cache is not None:
            self._total_cached_tokens -= leaf.token_count()
            leaf.kv_cache = None


# module-level singleton
cache = RadixCache()
