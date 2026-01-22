from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import json
import logging
from app.rag import rag_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class AskRequest(BaseModel):
    prompt: str

class AskResponse(BaseModel):
    response: str

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        logger.info(f"Received question: {request.prompt}")
        
        # 1. Retrieve context
        context, citations = rag_service.retrieve(request.prompt)
        logger.info(f"Context {context}, citations {citations} retrieved, calling LLM...")
        
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
        logger.info("LLM response received")
        cit = ' '.join(citations)
        result = response.json()
        model_response = result.get("response", "")
        final_response = model_response + "\n\nYou can refer: " + cit
        return AskResponse(response=final_response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
