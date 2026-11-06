"""Build the vector index. Run locally, then commit the `index/` folder.

    pip install -r requirements-ingest.txt
    python -m app.ingest                      # all PDFs in data/
    python -m app.ingest data/handbook.pdf    # specific files
"""
import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

from app import config
from app import embed as embed_mod

logging.getLogger("pypdf").setLevel(logging.ERROR)


def extract_text_pypdf(pdf_path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(filter(None, (p.extract_text() for p in PdfReader(pdf_path).pages)))


def extract_text_ocr(pdf_path: str) -> str:
    import pytesseract
    from pdf2image import convert_from_path

    return "\n".join(pytesseract.image_to_string(img) for img in convert_from_path(pdf_path))


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")  # PDFs are full of non-breaking spaces; they waste prompt tokens
    text = re.sub(r"\n[ \t\n]*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_text(pdf_path: str) -> str:
    text = extract_text_pypdf(pdf_path)
    if len(text.strip()) < 300:  # scanned PDF heuristic
        print(f"[OCR fallback] {pdf_path}")
        text = extract_text_ocr(pdf_path)
    return clean_text(text)


def chunk_text(text: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80).split_text(text)


def build_index(chunks: list[dict], out_dir: Path) -> int:
    """chunks: [{"text": ..., "source": ...}]. Overwrites the index in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = np.vstack([embed_mod.embed([c["text"] for c in chunks[i : i + 64]]) for i in range(0, len(chunks), 64)])
    np.save(out_dir / "embeddings.npy", vectors.astype(np.float32))
    (out_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="*", help="PDF files (default: data/*.pdf)")
    parser.add_argument("--out", default=str(config.INDEX_DIR))
    args = parser.parse_args()

    pdfs = args.pdfs or sorted(str(p) for p in (config.ROOT / "data").glob("*.pdf"))
    if not pdfs:
        raise SystemExit("No PDFs found.")

    chunks = []
    for pdf in pdfs:
        print(f"Ingesting: {pdf}")
        pieces = chunk_text(extract_text(pdf))
        chunks += [{"text": t, "source": Path(pdf).name} for t in pieces]
        print(f"  {len(pieces)} chunks")

    n = build_index(chunks, Path(args.out))
    print(f"Index written to {args.out} ({n} chunks)")


if __name__ == "__main__":
    main()
