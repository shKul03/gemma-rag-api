import logging
import os
import time
from collections import defaultdict, deque

import requests
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app import config, llm, rag

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("bakarwadi")

app = FastAPI(title="Bakarwadi", description="HR-policy RAG assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

NOT_FOUND = "NOT FOUND IN CONTEXT"

_hits: dict[str, deque] = defaultdict(deque)


def _rate_limited(request: Request) -> bool:
    if config.RATE_LIMIT_PER_MIN <= 0:
        return False
    fwd = request.headers.get("x-forwarded-for", "")
    ip = fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if not q:
        _hits.pop(ip, None)
        q = _hits[ip]
    if len(q) >= config.RATE_LIMIT_PER_MIN:
        return True
    q.append(now)
    return False

SYSTEM_PROMPT = f"""You are Bakarwadi, an HR policy assistant.

STRICT RULES:
- Answer ONLY using the context provided by the user message.
- If the answer is not present, reply exactly: {NOT_FOUND}
- Do not use outside knowledge.
- Do not guess."""


@app.get("/")
def root():
    return {"name": "Bakarwadi", "status": "ok", "endpoints": ["POST /ask", "GET /health"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: Request, question: str = Body(..., embed=True)):
    if _rate_limited(request):
        raise HTTPException(status_code=429, detail="Too many requests, slow down.")
    question = (question or "").strip()
    if not question:
        return {"answer": "Enter Non-Null Question"}
    if len(question) > config.MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail=f"Question too long (max {config.MAX_QUESTION_CHARS} characters)")

    try:
        hits = rag.retrieve(question)
    except FileNotFoundError:
        logger.exception("Index missing")
        raise HTTPException(status_code=500, detail="Knowledge index not found. Run `python -m app.ingest` and redeploy.")

    if not hits or hits[0]["score"] < config.MIN_SCORE:
        logger.info("No relevant context (best score %.3f)", hits[0]["score"] if hits else 0.0)
        return {"answer": NOT_FOUND, "sources": []}

    context = "\n\n".join(h["text"] for h in hits)
    user_msg = f"Context:\n{context}\n\nQuestion:\n{question}"

    try:
        answer = llm.chat(SYSTEM_PROMPT, user_msg)
    except llm.LLMNotConfigured:
        logger.error("LLM API key not configured")
        raise HTTPException(status_code=503, detail="Model is not configured on the server.")
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        logger.error("LLM provider returned %s", status)
        if status == 429:
            raise HTTPException(status_code=429, detail="Model rate limit reached, try again in a minute.")
        raise HTTPException(status_code=502, detail="The model provider returned an error.")
    except requests.RequestException:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail="Could not reach the model provider.")

    sources = sorted({os.path.basename(h["source"]) for h in hits})
    return {"answer": answer, "sources": sources}
