"""Tests for the SSE keepalive added to server.py's streaming generators:
when the token queue sits empty past KEEPALIVE_INTERVAL_SECONDS (e.g. a
low-priority request waiting behind a full batch), the generator must yield
an SSE comment line (`: keepalive\\n\\n`) instead of blocking forever, so the
client's read-timeout clock keeps getting reset. Once a real token/EOS
arrives, generation must proceed exactly as before."""

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

    def build_prompt(messages, tools=None, enable_thinking=False):
        raise NotImplementedError("not needed for these tests")

    fake.build_prompt = build_prompt
    sys.modules["inference"] = fake


_install_fake_inference()

import batched_inference as bi  # noqa: E402
import server  # noqa: E402


def make_seq(priority: str = "low") -> bi.Sequence:
    return bi.Sequence(
        input_ids=[0],
        max_tokens=10,
        temperature=0.7,
        top_p=0.9,
        priority=priority,
    )


@pytest.fixture(autouse=True)
def fast_keepalive(monkeypatch):
    """Shrink the keepalive interval so tests don't sleep for 15s."""
    monkeypatch.setattr(server, "KEEPALIVE_INTERVAL_SECONDS", 0.05)


def test_stream_sequence_emits_keepalive_while_queue_is_empty():
    seq = make_seq()
    gen = server.stream_sequence(seq, "req-1")

    # Nothing has been put on the queue yet -- the generator must time out
    # and yield a comment line rather than blocking indefinitely.
    chunk = next(gen)
    assert chunk == ": keepalive\n\n"


def test_stream_sequence_resumes_normally_after_keepalive():
    seq = make_seq()
    gen = server.stream_sequence(seq, "req-1")

    assert next(gen) == ": keepalive\n\n"

    # A real token (or EOS) arriving after a keepalive must be handled
    # exactly like the no-keepalive path: no extra keepalives, no data loss.
    seq.token_queue.put(None)  # EOS
    done_chunk = next(gen)
    assert done_chunk.startswith("data: ")
    assert '"finish_reason": "stop"' in done_chunk

    with pytest.raises(StopIteration):
        next(gen)


def test_stream_sequence_with_tools_emits_keepalive_while_queue_is_empty():
    seq = make_seq()
    gen = server.stream_sequence_with_tools(seq, "req-2")

    chunk = next(gen)
    assert chunk == ": keepalive\n\n"

    seq.token_queue.put(None)  # EOS, no tool calls
    done_chunk = next(gen)
    assert done_chunk.startswith("data: ")
    assert '"finish_reason": "stop"' in done_chunk


def test_keepalive_comment_line_is_ignored_by_the_sse_spec_prefix():
    seq = make_seq()
    gen = server.stream_sequence(seq, "req-3")
    chunk = next(gen)
    assert chunk.startswith(":")
    assert not chunk.startswith("data:")
