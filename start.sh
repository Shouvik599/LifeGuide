#!/bin/bash

# Check if the ChromaDB directory already exists
if [ ! -d "/code/chroma_db" ]; then
    echo "📦 ChromaDB not found. Starting ingestion..."
    python ingest.py
else
    echo "✅ ChromaDB found. Skipping ingestion."
fi

# Start the FastAPI application
echo "🚀 Starting FastAPI server..."
uvicorn app:app --host 0.0.0.0 --port 7860