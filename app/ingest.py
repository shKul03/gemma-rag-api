import os
import re
import json
import pandas as pd
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from supabase import create_client
import sys
from datetime import datetime
from pathlib import Path

# --- SQLite Fix for Linux/Server Environments ---
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
CHROMA_DIR = "chroma"
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
SUPABASE_URL = "https://edsvrikgfibjibcuhcpz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVkc3ZyaWtnZmliamliY3VoY3B6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTUyOTMwMiwiZXhwIjoyMDg1MTA1MzAyfQ.Kot3Srfitkzrc7Hp6WYqRKGReAFzRsqXWVoiWTCdk9U"

# ----------------------------------------

embedder = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------- SUPABASE SYNC ------------

def download_files_from_supabase():
    """Downloads all supported file types from Supabase storage."""
    response = supabase.storage.from_("rag_docs").list("hr_policies")
    
    # Define formats we can now process
    SUPPORTED_EXTENSIONS = (".pdf", ".csv", ".json", ".txt", ".md")
    
    # Ensure local data directory exists
    os.makedirs("../data", exist_ok=True)

    for r in response:
        name = r['name']
        if name.lower().endswith(SUPPORTED_EXTENSIONS):
            local_path = f"../data/{name}"
            try:
                with open(local_path, "wb+") as f:
                    file_data = supabase.storage.from_("rag_docs").download(f"hr_policies/{name}")
                    f.write(file_data)
                print(f"✅ {name} synced from Supabase")
            except Exception as e:
                print(f"❌ Error downloading {name}: {e}")

# ----------- EXTRACTION ENGINES ------------

def extract_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = [page.extract_text() for page in reader.pages if page.extract_text()]
    combined = "\n".join(text)
    
    # OCR fallback for scanned images
    if len(combined.strip()) < 300:
        print(f"[OCR fallback] {pdf_path}")
        images = convert_from_path(pdf_path)
        combined = "\n".join([pytesseract.image_to_string(img) for img in images])
    return combined

def extract_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    # Format: "Column1: Value | Column2: Value"
    return "\n".join(df.apply(lambda row: " | ".join([f"{col}: {val}" for col, val in row.items()]), axis=1))

def extract_json(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return "\n".join([json.dumps(item) for item in data])
    return json.dumps(data)

def extract_txt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def clean_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# ------------- CORE INGESTION -------------------

def ingest_file(file_path: str):
    """Processes a single file based on its extension with rich metadata."""
    ext = os.path.splitext(file_path)[1].lower()
    raw_text = ""

    if ext == ".pdf":
        raw_text = extract_pdf(file_path)
    elif ext == ".csv":
        raw_text = extract_csv(file_path)
    elif ext == ".json":
        raw_text = extract_json(file_path)
    elif ext in [".txt", ".md"]:
        raw_text = extract_txt(file_path)
    else:
        print(f"⚠️ Skipping unsupported format: {ext}")
        return 0

    text = clean_text(raw_text)
    if not text:
        return 0

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Get file metadata
    file_name = os.path.basename(file_path)
    file_stat = os.stat(file_path)
    last_modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
    file_size = file_stat.st_size
    
    # Create rich metadata for each chunk
    metadatas = []
    for i in range(len(chunks)):
        metadata = {
            'source': file_name,
            'chunk_index': i,
            'total_chunks': len(chunks),
            'last_updated': last_modified,
            'file_size': file_size,
            'file_type': ext.replace('.', ''),
            'ingestion_timestamp': datetime.utcnow().isoformat()
        }
        metadatas.append(metadata)

    # Embedding and Storing
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ Successfully added {len(chunks)} chunks from {file_name}")
    return len(chunks)

def ingest_data_folder(folder_path: str):
    """Iterates through the data folder and ingests everything."""
    if not os.path.isdir(folder_path):
        return

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if not f.startswith('.')]
    
    for f_path in files:
        try:
            ingest_file(f_path)
        except Exception as e:
            print(f"❌ Failed to ingest {f_path}: {e}")

if __name__ == "__main__":
    download_files_from_supabase()
    ingest_data_folder("../data")
    print(f"\nTotal vectors in Chroma DB: {collection.count()}")