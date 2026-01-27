FROM python:3.10-slim

# ---------------- SYSTEM DEPS ----------------
RUN apt-get update && apt-get install -y \
    curl \
    psmisc \
    zstd \
    tesseract-ocr \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# ---------------- OLLAMA ----------------
RUN curl -fsSL https://ollama.com/install.sh | sh

# ---------------- WORKDIR ----------------
WORKDIR /app

# ---------------- PYTHON DEPS ----------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------- APP CODE ----------------
COPY app/ ./app/

# ---------------- DATA DIRS ----------------
RUN mkdir -p /data /chroma

# ---------------- MODELS ----------------
RUN ollama serve & \
    sleep 10 && \
    ollama pull phi3:mini && \
    killall ollama

# ---------------- ENV ----------------
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV CHROMA_DIR=/chroma

# ---------------- API ----------------
EXPOSE 8080
CMD sh -c "ollama serve & sleep 10 && uvicorn app.main:app --host 0.0.0.0 --port 8080"
