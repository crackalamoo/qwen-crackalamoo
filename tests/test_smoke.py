"""Smoke test: confirms the test harness can import project modules (which
import mlx) and run."""

import mlx.core as mx


def test_mlx_works():
    assert mx.array([1, 2, 3]).sum().item() == 6


def test_imports():
    import prefix_cache  # noqa: F401
    import batch_quantized_cache  # noqa: F401
    import mlx_patches  # noqa: F401
