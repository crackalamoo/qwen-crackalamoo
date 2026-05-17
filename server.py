import json
import time
import uuid

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from batched_inference import submit_request, Sequence

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3-8b"
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    repetition_penalty: float = 1.1
    stream: bool = True


def make_chunk(content: str, request_id: str) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3-8b",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(payload)}\n\n"


def make_done_chunk(request_id: str) -> str:
    payload = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "qwen3-8b",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n"


def stream_sequence(seq: Sequence, request_id: str):
    """
    Sync generator: blocks on seq.token_queue.get() until tokens arrive.
    FastAPI runs this via iterate_in_threadpool so it doesn't stall the event loop.
    Local decoding state is kept here (not in Sequence) for thread safety.
    """
    local_ids: list[int] = []
    decoded_so_far = ""
    while True:
        token_id = seq.token_queue.get()
        if token_id is None:
            yield make_done_chunk(request_id)
            break
        local_ids.append(token_id)
        new_text = seq_decode(local_ids)
        safe_text = new_text.rstrip("�")
        if len(safe_text) > len(decoded_so_far):
            yield make_chunk(safe_text[len(decoded_so_far):], request_id)
            decoded_so_far = safe_text


def seq_decode(ids: list[int]) -> str:
    from batched_inference import tokenizer
    return tokenizer.decode(ids, skip_special_tokens=True)


@app.get("/chat")
async def chat_ui():
    return FileResponse(Path(__file__).parent / "static" / "chat.html")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    request_id = str(uuid.uuid4())
    messages = [m.model_dump() for m in request.messages]
    seq = submit_request(
        messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
    )
    return StreamingResponse(
        stream_sequence(seq, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
