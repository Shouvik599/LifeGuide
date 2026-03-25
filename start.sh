#!/bin/bash

# Use the absolute path relative to the app directory
CHROMA_PATH="./chroma_db"

if [ ! -d "$CHROMA_PATH" ]; then
    echo "📦 ChromaDB not found. Starting ingestion..."
    python ingest.py
else
    echo "✅ ChromaDB found. Skipping ingestion."
fi

echo "🚀 Starting FastAPI server with concurrency..."
# --workers 2 allows two simultaneous processes
# --timeout-keep-alive is increased for slow LLM responses
exec uvicorn app:app --host 0.0.0.0 --port 7860 --workers 2 --timeout-keep-alive 60