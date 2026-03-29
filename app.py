"""
app.py — FastAPI backend server for the Sacred Texts RAG application.

Endpoints:
    POST /ask          — Ask a question, get a streamed answer with sources
    POST /clear        — Clear conversation history for a session
    GET  /history      — Retrieve conversation history for a session
    GET  /health       — Health check
    GET  /books        — List books currently in the knowledge base

Run with:
    python app.py
"""

import os
import uuid
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from rag_chain import (
    query_sacred_texts,
    get_embeddings,
    get_vector_store,
    clear_session,
    get_history,
)
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sacred Texts RAG API",
    description="Ask questions answered exclusively from Bhagavad Gita, Quran, Bible, and Guru Granth Sahib",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Id"],
)

SESSION_COOKIE = "rag_session_id"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_or_create_session(request: Request, response: Response) -> str:
    """
    Read the session ID from the cookie (or X-Session-Id header).
    If absent, generate a new one and set it on the response cookie.
    """
    session_id = (
        request.cookies.get(SESSION_COOKIE)
        or request.headers.get("X-Session-Id")
    )
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(
            key=SESSION_COOKIE,
            value=session_id,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24,   # 24 hours
        )
    return session_id


# ─── Request / Response Models ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000,
                          example="What do the scriptures say about compassion?")
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversations. "
                    "If omitted, the server reads/creates one via cookie.",
    )

class HealthResponse(BaseModel):
    status: str
    message: str

class BooksResponse(BaseModel):
    books: list[str]
    total_chunks: int

class ClearRequest(BaseModel):
    session_id: str | None = None

class HistoryItem(BaseModel):
    role: str          # "human" | "ai"
    content: str

class HistoryResponse(BaseModel):
    session_id: str
    turns: int
    messages: list[HistoryItem]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return {"status": "ok", "message": "Sacred Texts RAG is running 🕊️"}


@app.get("/books", response_model=BooksResponse, tags=["Knowledge Base"])
def list_books():
    try:
        embeddings   = get_embeddings()
        vector_store = get_vector_store(embeddings)
        collection   = vector_store._collection
        results      = collection.get(include=["metadatas"])
        metadatas    = results.get("metadatas", [])
        books = sorted(set(m.get("book", "Unknown") for m in metadatas if m))
        return {"books": books, "total_chunks": len(metadatas)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read knowledge base: {e}")


@app.post("/ask", tags=["Query"])
async def ask(request_body: AskRequest, request: Request, response: Response):
    """
    Ask a spiritual or philosophical question.
    Streams the answer as NDJSON (one JSON object per line).
    Maintains per-session conversation history automatically via cookie or
    the `session_id` field in the request body.
    """
    if not request_body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Resolve session: body field > cookie/header > new
    if request_body.session_id:
        session_id = request_body.session_id
    else:
        session_id = get_or_create_session(request, response)

    try:
        stream = query_sacred_texts(request_body.question, session_id=session_id)

        # We need to forward the session_id so the frontend can persist it
        headers = {"X-Session-Id": session_id}

        return StreamingResponse(
            stream,
            media_type="application/x-ndjson",
            headers=headers,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not found. Run `python ingest.py` first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear", tags=["Session"])
async def clear_conversation(body: ClearRequest, request: Request, response: Response):
    """
    Clear the conversation history for the given session.
    If session_id is omitted, clears the session identified by cookie.
    """
    session_id = body.session_id or request.cookies.get(SESSION_COOKIE)
    if not session_id:
        raise HTTPException(status_code=400, detail="No session to clear.")
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/history", response_model=HistoryResponse, tags=["Session"])
async def conversation_history(session_id: str | None = None, request: Request = None):
    """
    Return the conversation history for a session (for debugging / display).
    """
    sid = session_id or (request.cookies.get(SESSION_COOKIE) if request else None)
    if not sid:
        raise HTTPException(status_code=400, detail="Provide session_id query param or cookie.")

    messages = get_history(sid)
    items = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            items.append(HistoryItem(role="human", content=msg.content))
        elif isinstance(msg, AIMessage):
            items.append(HistoryItem(role="ai", content=msg.content))

    return HistoryResponse(
        session_id=sid,
        turns=len(items) // 2,
        messages=items,
    )


@app.get("/", include_in_schema=False)
async def serve_frontend():
    frontend_path = "frontend/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "Sacred Texts RAG API is live. Visit /docs for Swagger UI."}


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))

    print(f"\n🕊️  Sacred Texts RAG — API Server v2.0")
    print(f"{'─' * 40}")
    print(f"🌐  Running at : http://{host}:{port}")
    print(f"🧠  Multi-turn conversation: ENABLED")
    print(f"{'─' * 40}\n")

    uvicorn.run("app:app", host=host, port=port, reload=False)