"""
app.py — FastAPI backend server for the Sacred Texts RAG application.

Endpoints:
    POST /ask          — Ask a question, get an answer with sources
    GET  /health       — Health check
    GET  /books        — List books currently in the knowledge base

Run with:
    python app.py
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_chain import query_sacred_texts, get_embeddings, get_vector_store  # ← FIXED

load_dotenv()

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sacred Texts RAG API",
    description="Ask questions answered exclusively from Bhagavad Gita, Quran, Bible, and Guru Granth Sahib",
    version="1.0.0",
)

# Allow requests from the local frontend (index.html opened as file://)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          example="What do the scriptures say about compassion?")

class Source(BaseModel):
    book: str
    page: int | str
    snippet: str

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]

class HealthResponse(BaseModel):
    status: str
    message: str

class BooksResponse(BaseModel):
    books: list[str]
    total_chunks: int


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check that the API is running."""
    return {"status": "ok", "message": "Sacred Texts RAG is running 🕊️"}


@app.get("/books", response_model=BooksResponse, tags=["Knowledge Base"])
def list_books():
    """List all books currently indexed in the knowledge base."""
    try:
        embeddings = get_embeddings()               # ← FIXED Step 1
        vector_store = get_vector_store(embeddings) # ← FIXED Step 2
        collection = vector_store._collection
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        books = sorted(set(
            m.get("book", "Unknown")
            for m in metadatas
            if m  # guard against None
        ))
        return {"books": books, "total_chunks": len(metadatas)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read knowledge base: {e}")


@app.post("/ask", response_model=AskResponse, tags=["Query"])
def ask(request: AskRequest):
    """
    Ask a spiritual or philosophical question.
    The answer is grounded strictly in the sacred texts.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = query_sacred_texts(request.question)
        return AskResponse(
            question=request.question,
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not found. Run `python ingest.py` first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serves the static frontend HTML file."""
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Sacred Texts RAG API is live. Visit /docs for Swagger UI."}

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # HF Spaces uses 7860 by default
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860")) 

    print(f"\n🕊️  Sacred Texts RAG — API Server")
    print(f"{'─' * 40}")
    print(f"🌐  Running at : http://{host}:{port}")
    print(f"{'─' * 40}\n")

    uvicorn.run("app:app", host=host, port=port, reload=False) # reload=False for production