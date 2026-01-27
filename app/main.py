from fastapi import FastAPI, Body, HTTPException
import requests

from app.rag import retrieve_context

# ---------------- APP ----------------

app = FastAPI()

# ---------------- OLLAMA CONFIG ----------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:mini"

# ---------------- LLM CALL ----------------

def call_phi3(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()

    result = response.json()
    return result.get("response", "").strip()

# ---------------- API ----------------

@app.post("/ask")
def ask(question: str = Body(..., embed=True)):
    try:
        # Retrieve RAG context
        context = retrieve_context(question)

        if not context.strip():
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
