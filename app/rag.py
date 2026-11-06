import json

import numpy as np

from app import config
from app import embed as embed_mod

_index = None


def _load():
    global _index
    if _index is None:
        emb = np.load(config.INDEX_DIR / "embeddings.npy")
        chunks = json.loads((config.INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        if len(chunks) != emb.shape[0]:
            raise RuntimeError("Index is corrupt: chunk count does not match embeddings")
        _index = (emb, chunks)
    return _index


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """Top-k chunks by cosine similarity (vectors are pre-normalised, so a dot product)."""
    emb, chunks = _load()
    q = embed_mod.embed([query])[0]
    scores = emb @ q
    top = np.argsort(-scores)[: (k or config.TOP_K)]
    return [
        {"text": chunks[i]["text"], "source": chunks[i]["source"], "score": float(scores[i])}
        for i in top
    ]
