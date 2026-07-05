"""Tests for the prompt-token ceiling in batched_inference.submit_request:
requests with a tokenized prompt over their effective limit must be rejected
with PromptTooLargeError before ever reaching the scheduler (avoiding the
Metal allocation crash on oversized prefill)."""

import sys
import types

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
    fake.build_prompt = None  # set per-test via monkeypatch on batched_inference's import
    sys.modules["inference"] = fake


_install_fake_inference()

import batched_inference as bi  # noqa: E402


class _FakeArray(list):
    def tolist(self):
        return list(self)


def _stub_build_prompt(n_tokens):
    """Return a fake `inference.build_prompt` yielding n_tokens ids."""
    def build_prompt(messages, tools=None, enable_thinking=False):
        return _FakeArray(range(n_tokens))
    return build_prompt


def _no_scheduler_submit(monkeypatch):
    """Fail loudly if the scheduler is ever touched -- used by the rejection
    tests, which must short-circuit before submission."""
    def boom(seq):
        raise AssertionError("scheduler.submit was called -- prompt-limit check did not short-circuit")
    monkeypatch.setattr(bi.scheduler, "submit", boom)


def test_prompt_under_hard_ceiling_passes_through(monkeypatch):
    monkeypatch.setattr(sys.modules["inference"], "build_prompt", _stub_build_prompt(100))
    monkeypatch.setattr(bi.scheduler, "submit", lambda seq: None)
    seq = bi.submit_request(messages=[{"role": "user", "content": "hi"}])
    assert len(seq.input_ids) == 100


def test_prompt_over_hard_ceiling_rejected_with_no_header(monkeypatch):
    monkeypatch.setattr(sys.modules["inference"], "build_prompt",
                         _stub_build_prompt(bi.HARD_MAX_PROMPT_TOKENS + 1))
    _no_scheduler_submit(monkeypatch)
    with pytest.raises(bi.PromptTooLargeError) as exc_info:
        bi.submit_request(messages=[{"role": "user", "content": "hi"}])
    assert exc_info.value.limit == bi.HARD_MAX_PROMPT_TOKENS
    assert exc_info.value.prompt_tokens == bi.HARD_MAX_PROMPT_TOKENS + 1


def test_header_tighter_than_hard_ceiling_rejects_at_header_value(monkeypatch):
    tight_limit = 1000
    monkeypatch.setattr(sys.modules["inference"], "build_prompt", _stub_build_prompt(tight_limit + 1))
    _no_scheduler_submit(monkeypatch)
    with pytest.raises(bi.PromptTooLargeError) as exc_info:
        bi.submit_request(messages=[{"role": "user", "content": "hi"}], max_prompt_tokens=tight_limit)
    assert exc_info.value.limit == tight_limit


def test_header_looser_than_hard_ceiling_still_rejects_at_hard_ceiling(monkeypatch):
    loose_header = bi.HARD_MAX_PROMPT_TOKENS * 10
    monkeypatch.setattr(sys.modules["inference"], "build_prompt",
                         _stub_build_prompt(bi.HARD_MAX_PROMPT_TOKENS + 1))
    _no_scheduler_submit(monkeypatch)
    with pytest.raises(bi.PromptTooLargeError) as exc_info:
        bi.submit_request(messages=[{"role": "user", "content": "hi"}], max_prompt_tokens=loose_header)
    assert exc_info.value.limit == bi.HARD_MAX_PROMPT_TOKENS


def test_prompt_within_tighter_header_passes_through(monkeypatch):
    monkeypatch.setattr(sys.modules["inference"], "build_prompt", _stub_build_prompt(500))
    monkeypatch.setattr(bi.scheduler, "submit", lambda seq: None)
    seq = bi.submit_request(messages=[{"role": "user", "content": "hi"}], max_prompt_tokens=1000)
    assert len(seq.input_ids) == 500
