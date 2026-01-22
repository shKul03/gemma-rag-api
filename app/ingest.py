import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Configuration
DATA_DIR = "data"
CHROMA_DIR = "chroma"
COLLECTION_NAME = "docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# CLIENT_ID = "35854dbb-ad3a-4fde-883b-336649ed8864"
# TENANT_ID = "7284c057-4078-4170-92c0-1bcb11c55269"
# CLIENT_SECRET = "qkX8Q~6An8pR-DzTbFsuQgUEzVcJ7Tmd-gPHhcCJ"


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks

def ingest_data():
    print("Starting ingestion...")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Create or get collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Initialize Embedding Model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Process PDFs
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} does not exist.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    if not files:
        print("No PDF files found.")
        return

    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        print(f"Processing {file}...")
        
        text = extract_text_from_pdf(file_path)
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        if not chunks:
            continue

        print(f"Generated {len(chunks)} chunks.")
        
        embeddings = model.encode(chunks).tolist()
        ids = [f"{file}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file} for _ in range(len(chunks))]

        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(chunks)} chunks to ChromaDB.")

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()
