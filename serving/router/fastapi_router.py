from __future__ import annotations
import os
import httpx
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

BACKEND_URL = os.getenv("BACKEND_URL", "http://host.docker.internal:8000/v1")
API_KEY = os.getenv("BACKEND_KEY", "devkey")

#  <--  Mistral is the default now
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-7b-base")


class ChatReq(BaseModel):
    user: str = Field(..., description="User prompt")


class ChatResp(BaseModel):
    content: str


app = FastAPI(title="Router", version="0.1.0")


@app.post("/chat", response_model=ChatResp)
async def chat(req: ChatReq):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": req.user}],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        try:
            r = await client.post(
                f"{BACKEND_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=str(e))
    return {"content": r.json()["choices"][0]["message"]["content"].strip()}
