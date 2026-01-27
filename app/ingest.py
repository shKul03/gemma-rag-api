import os
import re
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------
CHROMA_DIR = "chroma"
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ----------------------------------------

embedder = SentenceTransformer(EMBED_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


# ----------- TEXT EXTRACTION ------------

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


# if __name__ == "__main__":
#     ingest_pdf("data/sample_hr_policy.pdf")
#     print("Total vectors in DB:", collection.count())
