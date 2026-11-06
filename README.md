# Bakarwadi

A small HR-policy assistant. Ask a question, it finds the relevant passages in your HR handbook and has an LLM answer **only** from them (otherwise it replies `NOT FOUND IN CONTEXT`).

FastAPI · Groq (open models) · MiniLM embeddings via ONNX · prebuilt vector index · deploys on Vercel.

**Live:** https://bakarwadi.vercel.app (`GET /health`, `POST /ask`). The first request after idle takes about 8 seconds (cold start, model loads from the bundled `model/` folder); warm requests take 2 to 3 seconds. `/ask` is limited to 10 requests per minute per IP.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/ask` | `{"question": "How long is maternity leave?"}` | `{"answer": "...", "sources": ["handbook.pdf"]}` |
| GET | `/health` | | `{"status": "ok"}` |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # add your GROQ_API_KEY (free at console.groq.com)

python -m app.ingest        # builds index/ from data/*.pdf (first run downloads the embedding model)
uvicorn app.main:app --reload
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How many sick days do I get?"}'
```

Tests (no network or key needed): `python -m pytest`

## Use your own documents

Drop PDFs in `data/`, run `python -m app.ingest`, commit the regenerated `index/`, redeploy. Scanned PDFs fall back to OCR (needs Tesseract + Poppler installed locally; OCR only runs at ingest time).

## Configuration

| Variable | Default | |
|---|---|---|
| `GROQ_API_KEY` | – | required (or `LLM_API_KEY`) |
| `LLM_MODEL` | `openai/gpt-oss-120b` | any model your provider serves |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | any OpenAI-compatible endpoint |
| `MIN_SCORE` | `0.2` | below this similarity the LLM is not called |
| `ALLOWED_ORIGINS` | `*` | comma-separated CORS origins |
| `RATE_LIMIT_PER_MIN` | `10` | per-IP limit on `/ask` (best effort, per instance; `0` disables) |

## Deploy on Vercel

1. Push this repo to GitHub and import it at vercel.com/new (framework is detected as FastAPI).
2. Project Settings → Environment Variables → add `GROQ_API_KEY`.
3. Deploy, then check `/health` and POST to `/ask`.

The embedding model is downloaded to `/tmp` on a cold start, so the first request after idle is slower.

## Sample data and attribution

`data/sample_hr_policy.pdf` is a public nonprofit employee-handbook template, © 2004 National Council of Nonprofit Associations, reused with attribution. Replace it with your own PDFs and re-run `python -m app.ingest` to rebuild the index.
