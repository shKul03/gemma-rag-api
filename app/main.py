from fastapi import FastAPI, Body, HTTPException
import requests
import logging
from app.rag import retrieve_context

# --------------- Loggers ---------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('ShlokaLogsLoL')

# ---------------- APP ----------------

app = FastAPI()

# ---------------- OLLAMA CONFIG ----------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

# ---------------- LLM CALL ----------------

def call_phi3(prompt: str) -> str:
    logger.info(f"Calling SLM {MODEL_NAME} using Ollama")
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    result = response.json()
    logger.info(f"Received response from {MODEL_NAME}")
    return result.get("response", "").strip()

# ---------------- API ----------------

@app.post("/ask")
def ask(question: str = Body(..., embed=True)):
    if question:
        logger.info(f"Received ASK request")
    else:
        logger.warning("Question parameter is None type")
        return {"answer": "Enter Non-Null Question"}
    try:
        # Retrieve RAG context
        context = retrieve_context(question)
        logger.info("Context received")

        if not context.strip():
            logger.warning(f"No context received: {context}")
            return {"answer": "NOT FOUND IN CONTEXT"}

        # Build grounded prompt
        prompt = f"""
You are an HR policy assistant.

STRICT RULES:
- Answer ONLY using the context below.
- If the answer is not present, reply exactly: NOT FOUND IN CONTEXT.
- Do not use outside knowledge.
- Do not guess.

Context:
{context}

Question:
{question}
"""

        # Call Phi-3 via Ollama
        answer = call_phi3(prompt)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- HEALTH CHECK ----------------

@app.get("/health")
def health():
    return {"status": "ok"}
