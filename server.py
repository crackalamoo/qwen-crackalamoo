import json
import os
import time
import uuid

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from batched_inference import submit_request, Sequence

load_dotenv()

QWEN_PORT = int(os.environ.get("QWEN_PORT", "8000"))

app = FastAPI()


class Message(BaseModel):
    model_config = {"extra": "allow"}
    role: str
    content: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatRequest(BaseModel):
    model: str = "qwen3-8b"
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_completion_tokens: int = 512
    repetition_penalty: float = 1.1
    stream: bool = True
    tools: list | None = None
    stream_options: StreamOptions | None = None
    reasoning_effort: str = "minimal"  # minimal | low | medium | high


# minimal = thinking disabled entirely (no <think> tokens at all).
# low/medium/high = thinking enabled, hard-capped via forced "</think>" injection.
_THINKING_BUDGETS: dict[str, int | None] = {
    "low": 256,
    "medium": 512,
    "high": 1024,
}


def resolve_thinking(reasoning_effort: str) -> tuple[bool, int | None]:
    """Map a reasoning_effort level to (enable_thinking, thinking_budget)."""
    if reasoning_effort == "minimal":
        return False, None
    if reasoning_effort not in _THINKING_BUDGETS:
        raise ValueError(f"invalid reasoning_effort: {reasoning_effort!r}")
    return True, _THINKING_BUDGETS[reasoning_effort]


def make_chunk(content: str, request_id: str, include_usage: bool = False) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3-8b",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    if include_usage:
        payload["usage"] = None
    return f"data: {json.dumps(payload)}\n\n"


def make_reasoning_chunk(content: str, request_id: str, include_usage: bool = False) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3-8b",
        "choices": [{"index": 0, "delta": {"reasoning_content": content}, "finish_reason": None}],
    }
    if include_usage:
        payload["usage"] = None
    return f"data: {json.dumps(payload)}\n\n"


def make_done_chunk(request_id: str, finish_reason: str = "stop", include_usage: bool = False) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3-8b",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if include_usage:
        payload["usage"] = None
    return f"data: {json.dumps(payload)}\n\n"


def make_usage_chunk(request_id: str, usage: dict) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3-8b",
        "choices": [],
        "usage": usage,
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"


def build_usage(seq: Sequence, completion_tokens: int) -> dict:
    prompt_tokens = len(seq.input_ids)
    cached_tokens = seq.n_cached_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }


# ---------------------------------------------------------------------------
# Streaming tag state machine
# ---------------------------------------------------------------------------

_STATE_TEXT = "TEXT"
_STATE_MAYBE_TAG = "MAYBE_TAG"
_STATE_TOOL_CALL = "TOOL_CALL"
_STATE_THINK = "THINK"


class TagStateMachine:
    """Character-level state machine that parses streaming text and emits events.

    Events yielded:
        ("text", str)        — plain text to emit immediately
        ("tool_call", dict)  — parsed tool call: {"name": ..., "arguments": ...}
        ("reasoning", str)   — raw <think> block content to stream live
    """

    def __init__(self, enabled_tags: list[str]):
        """
        enabled_tags: list of opening tags to intercept, e.g. ["<tool_call>", "<think>"].
        Tags not in this list are passed through as plain text.
        """
        self._enabled_tags = enabled_tags
        self._state = _STATE_TEXT
        self._buffer = ""          # chars accumulated in MAYBE_TAG, TOOL_CALL, or THINK
        self._candidates: list[str] = []  # remaining candidate tags in MAYBE_TAG

    def feed(self, delta: str):
        """Feed a text delta; yield zero or more events.

        Accumulates consecutive ("text", ch) or ("reasoning", ch) results from
        self._process_char(ch) into single strings.
        """
        text = ""
        reasoning = ""
        for ch in delta:
            for event in self._process_char(ch):
                kind = event[0] if event else None
                if kind == "text":
                    if reasoning:
                        yield ("reasoning", reasoning)
                        reasoning = ""
                    text += event[1]
                elif kind == "reasoning":
                    if text:
                        yield ("text", text)
                        text = ""
                    reasoning += event[1]
                else:
                    if text:
                        yield ("text", text)
                        text = ""
                    if reasoning:
                        yield ("reasoning", reasoning)
                        reasoning = ""
                    yield event
        if text:
            yield ("text", text)
        if reasoning:
            yield ("reasoning", reasoning)

    def flush(self):
        """Signal end-of-stream; yield any remaining buffered events."""
        if self._state == _STATE_MAYBE_TAG:
            # Stray partial tag — emit as text
            if self._buffer:
                yield ("text", self._buffer)
            self._buffer = ""
        elif self._state == _STATE_TOOL_CALL:
            # Unclosed tool_call — drop it (matches old regex behavior)
            self._buffer = ""
        elif self._state == _STATE_THINK:
            # Unclosed think — emit any buffered partial content as reasoning
            if self._buffer:
                yield ("reasoning", self._buffer)
            self._buffer = ""
        # TEXT state: nothing buffered
        self._state = _STATE_TEXT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_char(self, ch: str):
        if self._state == _STATE_TEXT:
            if ch == "<":
                self._state = _STATE_MAYBE_TAG
                self._buffer = "<"
                self._candidates = list(self._enabled_tags)
            else:
                yield ("text", ch)

        elif self._state == _STATE_MAYBE_TAG:
            self._buffer += ch
            buf = self._buffer

            # Check for a full match first
            for tag in self._candidates:
                if buf == tag:
                    # Full match — transition to corresponding state
                    if tag == "<tool_call>":
                        self._state = _STATE_TOOL_CALL
                    elif tag == "<think>":
                        self._state = _STATE_THINK
                    self._buffer = ""
                    self._candidates = []
                    return  # nothing to yield

            # Filter candidates to those still compatible
            self._candidates = [t for t in self._candidates if t.startswith(buf)]

            if not self._candidates:
                # No candidate survived — flush buffer as text, back to TEXT
                flushed = self._buffer
                self._buffer = ""
                self._state = _STATE_TEXT
                yield ("text", flushed)

        elif self._state == _STATE_TOOL_CALL:
            self._buffer += ch
            if self._buffer.endswith("</tool_call>"):
                body = self._buffer[: -len("</tool_call>")].strip()
                self._buffer = ""
                self._state = _STATE_TEXT
                try:
                    data = json.loads(body)
                    args = data.get("arguments", {})
                    yield ("tool_call", {"name": data["name"], "arguments": args})
                except (json.JSONDecodeError, KeyError):
                    pass  # silently drop malformed tool call

        elif self._state == _STATE_THINK:
            self._buffer += ch
            closing = "</think>"
            if self._buffer == closing:
                # Closing tag fully matched — consume silently, back to TEXT
                self._buffer = ""
                self._state = _STATE_TEXT
            elif closing.startswith(self._buffer):
                # Still a potential prefix of </think> — keep buffering, don't yield yet
                pass
            else:
                # Not part of the closing tag — emit as reasoning content
                flushed = self._buffer
                self._buffer = ""
                yield ("reasoning", flushed)


# ---------------------------------------------------------------------------
# Stream generators
# ---------------------------------------------------------------------------

def stream_sequence_with_tools(seq: Sequence, request_id: str, include_usage: bool = False):
    """Real-time streaming with tag state machine — handles tool_call and think tags."""
    local_ids: list[int] = []
    decoded_so_far = ""

    parser = TagStateMachine(enabled_tags=["<tool_call>", "<think>"])
    tool_calls: list[dict] = []

    while True:
        token_id = seq.token_queue.get()
        if token_id is None:
            break
        local_ids.append(token_id)
        new_text = seq_decode(local_ids)
        safe_text = new_text.rstrip("�")
        if len(safe_text) > len(decoded_so_far):
            delta = safe_text[len(decoded_so_far):]
            decoded_so_far = safe_text
            for kind, value in parser.feed(delta):
                if kind == "text":
                    assert isinstance(value, str)
                    yield make_chunk(value, request_id, include_usage=include_usage)
                elif kind == "tool_call":
                    assert isinstance(value, dict)
                    tool_calls.append(value)
                elif kind == "reasoning":
                    assert isinstance(value, str)
                    yield make_reasoning_chunk(value, request_id, include_usage=include_usage)

    # EOS: flush any remaining buffer
    for kind, value in parser.flush():
        if kind == "text":
            assert isinstance(value, str)
            yield make_chunk(value, request_id, include_usage=include_usage)
        elif kind == "tool_call":
            assert isinstance(value, dict)
            tool_calls.append(value)
        elif kind == "reasoning":
            assert isinstance(value, str)
            yield make_reasoning_chunk(value, request_id, include_usage=include_usage)

    # Emit tool call chunks (after any text content, which is valid OpenAI format)
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            call_id = f"call_{uuid.uuid4().hex[:8]}"
            args_str = json.dumps(tc["arguments"]) if not isinstance(tc["arguments"], str) else tc["arguments"]
            name_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "qwen3-8b",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": i, "id": call_id, "type": "function", "function": {"name": tc["name"], "arguments": ""}}]
                }, "finish_reason": None}],
            }
            yield f"data: {json.dumps(name_chunk)}\n\n"
            args_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "qwen3-8b",
                "choices": [{"index": 0, "delta": {
                    "tool_calls": [{"index": i, "function": {"arguments": args_str}}]
                }, "finish_reason": None}],
            }
            yield f"data: {json.dumps(args_chunk)}\n\n"
        yield make_done_chunk(request_id, finish_reason="tool_calls", include_usage=include_usage)
    else:
        yield make_done_chunk(request_id, finish_reason="stop", include_usage=include_usage)

    if include_usage:
        yield make_usage_chunk(request_id, build_usage(seq, len(local_ids)))


def stream_sequence(seq: Sequence, request_id: str, include_usage: bool = False):
    """
    Sync generator: blocks on seq.token_queue.get() until tokens arrive.
    FastAPI runs this via iterate_in_threadpool so it doesn't stall the event loop.
    Local decoding state is kept here (not in Sequence) for thread safety.
    Suppresses <think>...</think> blocks in real time.
    """
    local_ids: list[int] = []
    decoded_so_far = ""
    parser = TagStateMachine(enabled_tags=["<think>"])

    while True:
        token_id = seq.token_queue.get()
        if token_id is None:
            for kind, value in parser.flush():
                if kind == "text":
                    assert isinstance(value, str)
                    yield make_chunk(value, request_id, include_usage=include_usage)
            yield make_done_chunk(request_id, finish_reason="stop", include_usage=include_usage)
            if include_usage:
                yield make_usage_chunk(request_id, build_usage(seq, len(local_ids)))
            break
        local_ids.append(token_id)
        new_text = seq_decode(local_ids)
        safe_text = new_text.rstrip("�")
        if len(safe_text) > len(decoded_so_far):
            delta = safe_text[len(decoded_so_far):]
            decoded_so_far = safe_text
            for kind, value in parser.feed(delta):
                if kind == "text":
                    assert isinstance(value, str)
                    yield make_chunk(value, request_id, include_usage=include_usage)
                # tool_call events won't occur since <tool_call> is not in enabled_tags


def collect_sequence(seq: Sequence, request_id: str) -> dict:
    """Non-streaming counterpart to stream_sequence. Drains the token queue and
    returns a complete chat.completion JSON object."""
    local_ids: list[int] = []
    decoded_so_far = ""
    content = ""
    parser = TagStateMachine(enabled_tags=["<think>"])

    while True:
        token_id = seq.token_queue.get()
        if token_id is None:
            for kind, value in parser.flush():
                if kind == "text":
                    content += value
            break
        local_ids.append(token_id)
        new_text = seq_decode(local_ids)
        safe_text = new_text.rstrip("�")
        if len(safe_text) > len(decoded_so_far):
            delta = safe_text[len(decoded_so_far):]
            decoded_so_far = safe_text
            for kind, value in parser.feed(delta):
                if kind == "text":
                    content += value

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen3-8b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "refusal": None}, "logprobs": None, "finish_reason": "stop"}],
        "usage": build_usage(seq, len(local_ids)),
    }


def collect_sequence_with_tools(seq: Sequence, request_id: str) -> dict:
    """Non-streaming counterpart to stream_sequence_with_tools."""
    local_ids: list[int] = []
    decoded_so_far = ""
    content = ""
    reasoning_content = ""
    tool_calls: list[dict] = []
    parser = TagStateMachine(enabled_tags=["<tool_call>", "<think>"])

    while True:
        token_id = seq.token_queue.get()
        if token_id is None:
            break
        local_ids.append(token_id)
        new_text = seq_decode(local_ids)
        safe_text = new_text.rstrip("�")
        if len(safe_text) > len(decoded_so_far):
            delta = safe_text[len(decoded_so_far):]
            decoded_so_far = safe_text
            for kind, value in parser.feed(delta):
                if kind == "text":
                    content += value
                elif kind == "reasoning":
                    reasoning_content += value
                elif kind == "tool_call":
                    tool_calls.append(value)

    for kind, value in parser.flush():
        if kind == "text":
            content += value
        elif kind == "reasoning":
            reasoning_content += value
        elif kind == "tool_call":
            tool_calls.append(value)

    finish_reason = "tool_calls" if tool_calls else "stop"
    message: dict = {"role": "assistant", "content": content or None, "refusal": None}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]) if not isinstance(tc["arguments"], str) else tc["arguments"],
                },
            }
            for tc in tool_calls
        ]

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen3-8b",
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish_reason}],
        "usage": build_usage(seq, len(local_ids)),
    }


def seq_decode(ids: list[int]) -> str:
    from batched_inference import tokenizer
    return tokenizer.decode(ids, skip_special_tokens=True)


@app.get("/chat")
async def chat_ui():
    return FileResponse(Path(__file__).parent / "static" / "chat.html")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    request_id = str(uuid.uuid4())
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    enable_thinking, thinking_budget = resolve_thinking(request.reasoning_effort)
    seq = submit_request(
        messages,
        max_tokens=request.max_completion_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )
    if not request.stream:
        result = (
            collect_sequence_with_tools(seq, request_id)
            if request.tools
            else collect_sequence(seq, request_id)
        )
        return JSONResponse(content=result)

    include_usage = bool(request.stream_options and request.stream_options.include_usage)
    generator = (
        stream_sequence_with_tools(seq, request_id, include_usage=include_usage)
        if request.tools
        else stream_sequence(seq, request_id, include_usage=include_usage)
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=QWEN_PORT)

