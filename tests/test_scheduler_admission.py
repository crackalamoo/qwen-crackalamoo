"""Tests for memory-aware, priority-aware admission in BatchScheduler
(batched_inference.py). Stubs `inference` (which otherwise loads the real
Qwen3-8B model at import time) before importing batched_inference."""

import sys
import types
import time

import pytest


def _install_fake_inference():
    if "inference" in sys.modules:
        return
    fake = types.ModuleType("inference")
    fake.model = object()

    class FakeTokenizer:
        eos_token_id = -1

        def decode(self, ids, skip_special_tokens=True):
            return ""

    fake.tokenizer = FakeTokenizer()
    fake.sample = lambda logits, temperature, top_p: None

    def build_prompt(messages, tools=None, enable_thinking=False):
        raise NotImplementedError("not needed for these tests")

    fake.build_prompt = build_prompt
    sys.modules["inference"] = fake


_install_fake_inference()

import batched_inference as bi  # noqa: E402


def make_seq(prompt_tokens: int, max_tokens: int = 100, priority: str = "high") -> bi.Sequence:
    return bi.Sequence(
        input_ids=[0] * prompt_tokens,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        priority=priority,
    )


@pytest.fixture
def scheduler_shell():
    """A BatchScheduler-like object with the real admission methods but no
    background thread and no active[] contents -- avoids touching the model."""
    sched = bi.BatchScheduler.__new__(bi.BatchScheduler)
    sched.pending = bi.PendingQueue()
    sched.active = []
    sched._batch_cache = None
    sched._tick = 0
    return sched


# ---------------------------------------------------------------------------
# Part 1: memory-aware admission
# ---------------------------------------------------------------------------

def test_kv_cost_bytes_matches_formula():
    seq = make_seq(prompt_tokens=100, max_tokens=50)
    assert bi.kv_cost_bytes(seq) == (100 + 50) * bi.KV_BYTES_PER_TOKEN


def test_fits_budget_true_when_room_available(scheduler_shell):
    small = make_seq(prompt_tokens=10, max_tokens=10)
    assert scheduler_shell._fits_budget(small) is True


def test_fits_budget_false_when_candidate_alone_exceeds_budget(scheduler_shell):
    huge_tokens = bi.ACTIVE_KV_BUDGET_BYTES // bi.KV_BYTES_PER_TOKEN + 1000
    huge = make_seq(prompt_tokens=huge_tokens, max_tokens=0)
    assert scheduler_shell._fits_budget(huge) is False


def test_fits_budget_accounts_for_already_active_sequences(scheduler_shell):
    # fill active up close to the budget
    filler_tokens = bi.ACTIVE_KV_BUDGET_BYTES // bi.KV_BYTES_PER_TOKEN - 10
    scheduler_shell.active.append(make_seq(prompt_tokens=filler_tokens, max_tokens=0))

    tiny = make_seq(prompt_tokens=1, max_tokens=1)
    still_fits = scheduler_shell._fits_budget(tiny)

    bigger = make_seq(prompt_tokens=1000, max_tokens=1000)
    no_longer_fits = scheduler_shell._fits_budget(bigger)

    assert still_fits is True
    assert no_longer_fits is False


def test_try_pop_best_defers_oversized_candidate_instead_of_erroring(scheduler_shell):
    """A too-large sequence is left queued (retried later), never rejected."""
    huge_tokens = bi.ACTIVE_KV_BUDGET_BYTES // bi.KV_BYTES_PER_TOKEN + 1000
    huge = make_seq(prompt_tokens=huge_tokens, max_tokens=0)
    scheduler_shell.pending.put(huge)

    popped = scheduler_shell.pending.try_pop_best(scheduler_shell._fits_budget)

    assert popped is None
    assert len(scheduler_shell.pending) == 1  # still queued, not dropped


def test_try_pop_best_skips_oversized_head_and_admits_smaller_one_behind_it(scheduler_shell):
    huge_tokens = bi.ACTIVE_KV_BUDGET_BYTES // bi.KV_BYTES_PER_TOKEN + 1000
    huge = make_seq(prompt_tokens=huge_tokens, max_tokens=0, priority="high")
    small = make_seq(prompt_tokens=10, max_tokens=10, priority="low")

    scheduler_shell.pending.put(huge)
    scheduler_shell.pending.put(small)

    popped = scheduler_shell.pending.try_pop_best(scheduler_shell._fits_budget)

    assert popped is small
    assert len(scheduler_shell.pending) == 1


def test_pop_best_blocking_admits_even_oversized_when_nothing_active(scheduler_shell):
    """The active-empty override: nothing can ever free memory if the
    scheduler refuses to admit anything, so this path ignores the budget."""
    huge_tokens = bi.ACTIVE_KV_BUDGET_BYTES // bi.KV_BYTES_PER_TOKEN + 1000
    huge = make_seq(prompt_tokens=huge_tokens, max_tokens=0)
    scheduler_shell.pending.put(huge)

    popped = scheduler_shell.pending.pop_best_blocking()

    assert popped is huge


# ---------------------------------------------------------------------------
# Part 2: priority + aging
# ---------------------------------------------------------------------------

def test_high_priority_preferred_over_low_when_both_fit(scheduler_shell):
    now = time.monotonic()
    low = make_seq(prompt_tokens=10, max_tokens=10, priority="low")
    high = make_seq(prompt_tokens=10, max_tokens=10, priority="high")
    low.enqueue_time = now
    high.enqueue_time = now  # arrived at the same instant -- no aging edge

    scheduler_shell.pending.put(low)
    scheduler_shell.pending.put(high)

    popped = scheduler_shell.pending.try_pop_best(scheduler_shell._fits_budget)

    assert popped is high


def test_aging_lets_long_waiting_low_priority_beat_fresh_high_priority(scheduler_shell):
    now = time.monotonic()
    old_low = make_seq(prompt_tokens=10, max_tokens=10, priority="low")
    fresh_high = make_seq(prompt_tokens=10, max_tokens=10, priority="high")

    scheduler_shell.pending.put(old_low)
    scheduler_shell.pending.put(fresh_high)

    # Simulate old_low having waited past the starvation bound T = (base_high
    # - base_low) / aging_rate, and fresh_high just having arrived.
    t = (bi.BASE_PRIORITY["high"] - bi.BASE_PRIORITY["low"]) / bi.AGING_RATE
    old_low.enqueue_time = now - (t + 1.0)
    fresh_high.enqueue_time = now

    popped = scheduler_shell.pending.try_pop_best(scheduler_shell._fits_budget)

    assert popped is old_low


def test_effective_priority_ages_linearly():
    seq = make_seq(prompt_tokens=1, priority="low")
    seq.enqueue_time = 0.0
    now = 5.0
    expected = bi.BASE_PRIORITY["low"] + 5.0 * bi.AGING_RATE
    assert bi.effective_priority(seq, now) == pytest.approx(expected)
