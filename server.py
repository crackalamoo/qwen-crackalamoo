import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference import generate
from inference import tokenizer

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


def stream_response(request: ChatRequest):
    request_id = str(uuid.uuid4())
    messages = [m.model_dump() for m in request.messages]
    for token in generate(messages, max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p):
        yield make_chunk(token, request_id)
    yield make_done_chunk(request_id)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    return StreamingResponse(
        stream_response(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

