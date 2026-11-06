import numpy as np

from app import config

_model = None


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings, shape (len(texts), dim)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding  # imported lazily so tests/tools can stub this module

        kwargs = {}
        if (config.MODEL_DIR / "model.onnx").is_file():
            kwargs["specific_model_path"] = str(config.MODEL_DIR)  # bundled copy: no network at cold start
        _model = TextEmbedding(model_name=config.EMBED_MODEL, cache_dir=config.EMBED_CACHE_DIR, **kwargs)
    vecs = np.array(list(_model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
