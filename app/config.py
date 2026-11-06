"""Runtime settings, all overridable through environment variables."""
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Tiny .env loader for local dev (no extra dependency). Real environment variables win.
    Tolerates `key = value`, lowercase names, quotes and comments."""
    f = ROOT / ".env"
    if not f.is_file():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip().upper(), v.strip().strip("\"'"))


_load_dotenv()

# Prebuilt vector index (created by `python -m app.ingest`, committed to the repo)
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "index"))

# Embeddings run locally through ONNX (no torch). Must match the model used at ingest time.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", os.path.join(tempfile.gettempdir(), "fastembed_cache"))

# Any OpenAI-compatible chat endpoint works. Defaults to Groq.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

TOP_K = int(os.getenv("TOP_K", "5"))
# If the best chunk scores below this, skip the LLM call and answer NOT FOUND.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.2"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "500"))
# Best-effort per-IP limit on /ask (in-memory, so it is per serverless instance). 0 disables.
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]


def llm_api_key() -> str:
    return os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or ""
