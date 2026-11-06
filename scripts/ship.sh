#!/usr/bin/env bash
# One-shot: install deps, run tests, build the real index, smoke-test against Groq.
#   bash scripts/ship.sh            # build + test
#   bash scripts/ship.sh --commit   # ...and commit the result (does not push)
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing dependencies"
pip install -q -r requirements-dev.txt

echo "==> Unit tests"
python -m pytest -q

echo "==> Building index from data/*.pdf (first run downloads the embedding model)"
python -m app.ingest

echo "==> Starting server and smoke-testing against Groq"
PORT=8765
uvicorn app.main:app --port "$PORT" --log-level warning &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf "localhost:$PORT/health" >/dev/null && break
  sleep 1
done
curl -sf "localhost:$PORT/health" >/dev/null || { echo "Server did not start"; exit 1; }

ask() {
  echo; echo "Q: $1"
  curl -sf -m 90 "localhost:$PORT/ask" -H 'content-type: application/json' \
    -d "{\"question\": \"$1\"}" || { echo "  REQUEST FAILED (check GROQ_API_KEY / rate limit)"; return 1; }
  echo
}
FAILED=0
ask "What is the personal leave policy?" || FAILED=1
ask "What does the handbook say about sexual harassment?" || FAILED=1
ask "What is the capital of France?" || FAILED=1   # should answer NOT FOUND IN CONTEXT

[ "$FAILED" -eq 0 ] || { echo; echo "Smoke test failed, not committing."; exit 1; }
echo; echo "==> Smoke test passed. index/ is ready."

if [ "${1:-}" = "--commit" ]; then
  git add app tests scripts index requirements.txt requirements-ingest.txt requirements-dev.txt \
          pyproject.toml vercel.json README.md about.txt .env.example .gitignore
  git commit -m "Switch to Groq and prebuilt index for Vercel"
  echo "==> Committed. Push with: git push"
else
  echo "Next: bash scripts/ship.sh --commit   (or commit manually), then git push and import on Vercel."
fi
