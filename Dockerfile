# FROM python:3.10-slim

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     curl \
#     psmisc \
#     && rm -rf /var/lib/apt/lists/*

# # Install Ollama
# RUN curl -fsSL https://ollama.com/install.sh | sh

# # Set working directory
# WORKDIR /app

# # Install Python dependencies
# COPY app/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY app/ ./app/

# # Create data directories
# RUN mkdir -p /data /chroma

# # Pull Gemma 2b model during build
# # We need to start ollama server in background, pull model, then stop it
# RUN ollama serve & \
#     sleep 5 && \
#     ollama pull gemma:2b && \
#     killall ollama

# # Expose port
# EXPOSE 8080

# # Environment variables
# ENV PYTHONUNBUFFERED=1
# ENV PORT=8080

# # Create start script
# RUN echo '#!/bin/bash' > start.sh && \
#     echo 'ollama serve &' >> start.sh && \
#     echo 'sleep 5' >> start.sh && \
#     echo 'if [ -f /app/app/ingest.py ]; then python /app/app/ingest.py; fi' >> start.sh && \
#     echo 'uvicorn app.main:app --host 0.0.0.0 --port 8080' >> start.sh && \
#     chmod +x start.sh

# CMD ["./start.sh"]

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    psmisc \
    zstd \
 && rm -rf /var/lib/apt/lists/*


# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy PDF data
COPY data/ /data/

# Create persistent vector directory
RUN mkdir -p /chroma

# Pull Gemma 2B model during build
RUN ollama serve & \
    sleep 10 && \
    ollama pull phi3:mini && \
    killall ollama

# Run ingestion ONCE at build time
RUN python app/ingest.py

# Expose API port
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Start Ollama and FastAPI
CMD sh -c "ollama serve & sleep 10 && uvicorn app.main:app --host 0.0.0.0 --port 8080"
