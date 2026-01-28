import os
import re
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from supabase import create_client
import tempfile
import sys

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
UPABASE_URL = "https://edsvrikgfibjibcuhcpz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVkc3ZyaWtnZmliamliY3VoY3B6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTUyOTMwMiwiZXhwIjoyMDg1MTA1MzAyfQ.Kot3Srfitkzrc7Hp6WYqRKGReAFzRsqXWVoiWTCdk9U"

# ----------------------------------------

embedder = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_NAME = "rag_docs"


# ----------- TEXT EXTRACTION ------------

def download_files_from_supabase():
    response = (
        supabase.storage
        .from_("rag_docs")
        .list(
            "hr_policies",
            {
                "limit": 100,
                "offset": 0,
                "sortBy": {"column": "name", "order": "desc"},
            }
        )
    )

    names = []
    for r in response:
        n = r['name']
        if n.endswith("pdf"):
            print(n)
            names.append(n)
    for name in names:
        with open(f"../data/{name}", "wb+") as f:
            pdf = supabase.storage.from_("rag_docs").download(f"hr_policies/{name}")
            f.write(pdf)
            print(f"{name} downloaded successfully")


def extract_text_pypdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = []

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text.append(t)

    return "\n".join(text)


def extract_text_ocr(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    text = []

    for img in images:
        text.append(pytesseract.image_to_string(img))

    return "\n".join(text)


# Collapse multiple whitespace/newline characters into a single character
def clean_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_text(pdf_path: str) -> str:
    text = extract_text_pypdf(pdf_path)

    # fallback heuristic
    if len(text.strip()) < 300:
        print(f"[OCR fallback] {pdf_path}")
        text = extract_text_ocr(pdf_path)

    return clean_text(text)


# ------------- CHUNKING -----------------

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    return splitter.split_text(text)


# ------------- INGEST -------------------

def ingest_pdf(pdf_path: str):
    print(f"Ingesting: {pdf_path}")

    text = extract_text(pdf_path)
    if not text:
        print("❌ No text extracted, skipping")
        return

    chunks = chunk_text(text)
    if not chunks:
        print("❌ No chunks created, skipping")
        return

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": pdf_path}] * len(chunks)
    )

    print(f"✅ Added {len(chunks)} chunks")


def ingest_pdf_folder(folder_path: str):
    if not os.path.isdir(folder_path):
        print(f"❌ Not a directory: {folder_path}")
        return

    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("⚠️ No PDF files found in folder")
        return

    print(f"📂 Found {len(pdf_files)} PDFs. Starting ingestion...\n")

    for pdf_path in pdf_files:
        try:
            ingest_pdf(pdf_path)
        except Exception as e:
            print(f"❌ Failed to ingest {pdf_path}: {e}")

    print("\n🎉 Folder ingestion completed")


if __name__ == "__main__":
    download_files_from_supabase()
    ingest_pdf_folder("data")
    print("Total vectors in DB:", collection.count())
