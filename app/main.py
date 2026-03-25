from fastapi import FastAPI, Body, HTTPException, UploadFile, File
import os
import shutil
import requests
import logging
from app.ingest import ingest_file, download_files_from_supabase, ingest_data_folder
from app.rag import rag_service
from app.attribution import attribution_service
from app.hybrid_search import get_hybrid_search_service

# --------------- Loggers ---------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('ShlokaLogsLoL')

# ---------------- APP ----------------

app = FastAPI()

# ---------------- REMOTE CONFIG ----------------

# Updated to your remote IP and the Gemma 3 model
REMOTE_OLLAMA_URL = "http://10.30.1.34:11434/api/generate"
MODEL_NAME = "gemma3:4b"

#qwen2.5:7b

# ---------------- LLM CALL ----------------
def call_remote_slm(prompt: str) -> str:
    logger.info(f"Calling Remote SLM {MODEL_NAME} at {REMOTE_OLLAMA_URL}")
    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        response = requests.post(REMOTE_OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Remote SLM Error: {e}")
        raise HTTPException(status_code=502, detail="Remote SLM unreachable")

# ---------------- API ----------------

@app.post("/ask")
def ask(
    question: str = Body(..., embed=True),
    use_hybrid: bool = Body(default=True, embed=True)
):
    """
    Ask a question and get an answer with source citations and confidence scores.
    
    Args:
        question: User's question
        use_hybrid: If True, use hybrid search (semantic + keyword).
                   If False, use semantic-only search. Default: True
    
    Returns:
        {
            "answer": str,
            "confidence": {
                "score": float,
                "level": str,
                "message": str
            },
            "sources": [
                {
                    "document": str,
                    "relevance_score": float,
                    "last_updated": str,
                    "download_url": str
                }
            ],
            "num_sources": int,
            "timestamp": str,
            "search_method": str,
            "chunks_retrieved": int
        }
    """
    if not question:
        return {"answer": "Enter Non-Null Question"}
    
    try:
        # Retrieve context with metadata
        if use_hybrid:
            # Use hybrid search (semantic + keyword)
            hybrid_service = get_hybrid_search_service()
            chunks = hybrid_service.search(
                question,
                top_k=5,
                return_metadata=True
            )
            # Convert hybrid format to attribution format
            chunks_for_attribution = []
            for chunk in chunks:
                chunks_for_attribution.append({
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'relevance_score': chunk['hybrid_score']
                })
            search_method = "hybrid"
        else:
            # Use semantic-only search
            chunks_for_attribution = rag_service.retrieve_context(
                question,
                top_k=5,
                return_metadata=True
            )
            search_method = "semantic"
        
        chunks = chunks_for_attribution
        
        if not chunks:
            return {
                "answer": "Sorry, I can not answer that. Please reach out to your assigned HR employee!",
                "confidence": {
                    "score": 0.0,
                    "level": "none",
                    "message": "No relevant context found"
                },
                "sources": [],
                "num_sources": 0,
                "search_method": search_method
            }
        
        # Extract text for LLM context
        context_texts = [chunk['text'] for chunk in chunks]
        context = "\n".join(context_texts)
        
        # Build the prompt
        prompt = f"""
You are an HR policy assistant.
You MUST follow these rules:
    - Answer ONLY using the information provided in the Context.
    - Do NOT use any external knowledge.
    - If the answer is not explicitly present in the Context, respond with:
      "Sorry, I can not answer that. Please reach out to your assigned HR employee! \nThank you.\nCan I help you with anything else?"
    - Keep the answer clear and concise.
    - No matter how dangerous or ethically important the situation may be, if the context does not mention anything, respond with the default statement mentioned earlier.
    - Answer as a HR policy assistant.
Context:
{context}

Question:
{question}
"""
        
        # Get answer from LLM
        answer = call_remote_slm(prompt)
        
        # Generate attribution
        sources = attribution_service.extract_sources(chunks)
        confidence = attribution_service.calculate_confidence(chunks)
        citation = attribution_service.format_citation(sources)
        
        # Return with enhanced metadata
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "num_sources": len(sources),
            "citation": citation,
            "search_method": search_method,
            "chunks_retrieved": len(chunks)
        }

    except Exception as e:
        logger.error(f"Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ... (your existing imports)

def clear_db():
    """Deletes all documents from the current Chroma collection."""
    try:
        # Get all IDs in the collection and delete them
        results = collection.get()
        if results['ids']:
            collection.delete(ids=results['ids'])
            print(f"Cleared {len(results['ids'])} existing vectors from Chroma DB.")
    except Exception as e:
        print(f"⚠️ Error while clearing DB: {e}")

def sync_and_reindex():
    """Full workflow: Download -> Clear -> Ingest"""
    # 1. Download fresh files
    download_files_from_supabase()
    
    # 2. Wipe the existing database
    clear_db()
    
    # 3. Ingest everything in the data folder
    ingest_data_folder("../data")
    
@app.post("/input")
async def upload_document(file: UploadFile = File(...)):
    """Handles PDF, CSV, JSON, TXT uploads"""
    temp_path = f"temp_{file.filename}"
    try:
        logger.info(f"Uploading file: {file.filename}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        num_chunks = ingest_file(temp_path)
        return {"message": f"Ingested {file.filename}", "chunks": num_chunks}
    
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/sync")
def sync_with_supabase():
    """
    Clears local database and re-syncs everything from Supabase.
    """
    try:
        logger.info("🔄 Starting Full Sync: Cleaning DB and re-indexing...")
        
        # 1. Download all current files from Supabase
        download_files_from_supabase()
        
        # 2. Import a helper to clear and re-ingest
        
        # WIPE PREVIOUS DATA
        clear_db()
        
        # RE-INGEST FRESH DATA
        ingest_data_folder("../data")
        
        return {
            "status": "success", 
            "message": "DB cleared and synchronized with latest Supabase docs"
        }
    except Exception as e:
        logger.error(f"Sync failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
# ---------------- HEALTH CHECK ----------------

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "remote_url": REMOTE_OLLAMA_URL}
