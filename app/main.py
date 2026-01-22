from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
from app.rag import rag_service

app = FastAPI()

class AskRequest(BaseModel):
    prompt: str

class AskResponse(BaseModel):
    response: str

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        # 1. Retrieve context
        context, citations = rag_service.retrieve(request.prompt)
        
        # 2. Format prompt
        final_prompt = rag_service.format_prompt(request.prompt, context)
        
        # 3. Call Ollama
        payload = {
            "model": "phi3:mini",
            "prompt": final_prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        cit = ' '.join(citations)
        result = response.json()
        model_response = result.get("response", "")
        final_response = model_response + "\n\nYou can refer: " + cit
        return AskResponse(response=final_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
