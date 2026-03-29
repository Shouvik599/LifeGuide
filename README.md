---
title: Sacred Texts RAG
emoji: 🕊️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# 🕊️ Sacred Texts RAG — Multi-Religion Knowledge Base

A Retrieval-Augmented Generation (RAG) application that answers spiritual queries using the Bhagavad Gita, Quran, Bible, and Guru Granth Sahib as the sole knowledge sources. Now with **multi-turn conversation memory** — ask follow-up questions naturally, just like a real dialogue.

---

## 📁 Project Structure

```
sacred-texts-rag/
├── README.md
├── requirements.txt
├── .env.example
├── ingest.py               # Step 1: Load PDFs → chunk → embed → store
├── rag_chain.py            # Core RAG chain logic (with session memory)
├── app.py                  # FastAPI backend server
└── frontend/
    └── index.html          # Chat UI (served by FastAPI)
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your NVIDIA_API_KEY
```

### 3. Add Your PDF Books
Place your PDF files in a `books/` folder:
```
books/
├── bhagavad_gita.pdf
├── quran.pdf
├── bible.pdf
└── guru_granth_sahib.pdf
```

### 4. Ingest the Books (Run Once)
```bash
python ingest.py
```
This will:
- Load and parse all PDFs
- Split into semantic chunks
- Create embeddings using NVIDIA's `llama-nemotron-embed-vl-1b-v2` model
- Store in a local ChromaDB vector store (`./chroma_db/`)

### 5. Start the Backend
```bash
python app.py
```
Server runs at: `http://localhost:7860`

### 6. Open the Frontend
Navigate to `http://localhost:7860` in your browser — the FastAPI server serves the UI directly.

---

## 🔑 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `NVIDIA_API_KEY` | Your NVIDIA API key | — |
| `CHROMA_DB_PATH` | Path to ChromaDB storage | `./chroma_db` |
| `COLLECTION_NAME` | ChromaDB collection name | `sacred_texts` |
| `CHUNKS_PER_BOOK` | Chunks retrieved per book per query | `3` |
| `MAX_HISTORY_TURNS` | Max conversation turns kept in memory per session | `6` |
| `HOST` | Server bind host | `0.0.0.0` |
| `PORT` | Server port | `7860` |

---

## 🧠 How It Works

```
User Query
    │
    ▼
[Session Memory]  ←── Injects prior conversation turns into LLM context
    │
    ▼
[Query Augmentation]  ←── Short follow-ups are enriched with previous question
    │
    ▼
[Hybrid Retrieval: BM25 + Vector Search]  ←── Per-book guaranteed slots
    │
    ▼
[NVIDIA Reranker]  ←── llama-3.2-nv-rerankqa-1b-v2 re-scores pooled candidates
    │
    ▼
[Semantic Cache Check]  ←── Skip LLM if a similar question was answered before
    │
    ▼
[Prompt with Context + History]
    │
    ▼
[Llama-3.3-70b-instruct]  ←── Answer grounded ONLY in retrieved texts
    │
    ▼
Streamed response with source citations (book + chapter/verse)
```

---

## 💬 Multi-Turn Conversation

The app maintains per-session conversation history so you can ask natural follow-up questions:

```
You:  "What do the scriptures say about forgiveness?"
AI:   [Answer citing Gita, Quran, Bible, Guru Granth Sahib]

You:  "Elaborate on the second point"       ← follow-up, no context needed
AI:   [Continues from previous answer]

You:  "What does the Bible say specifically?"  ← drill-down
AI:   [Focuses on Bible passages from the thread]
```

**How sessions work:**
- A session ID is created automatically on your first question and persisted in the browser's `localStorage`
- The server keeps the last `MAX_HISTORY_TURNS` (default: 6) human+AI pairs in memory
- Click **↺ New Conversation** in the header to clear history and start fresh
- Sessions are scoped to the server process — they reset on server restart

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ask` | Ask a question; streams NDJSON response |
| `POST` | `/clear` | Clear conversation history for a session |
| `GET` | `/history` | Inspect conversation history for a session |
| `GET` | `/books` | List all books indexed in the knowledge base |
| `GET` | `/health` | Health check |
| `GET` | `/` | Serves the frontend UI |
| `GET` | `/docs` | Swagger UI |

### `/ask` Request Body
```json
{
  "question": "What do the scriptures say about compassion?",
  "session_id": "optional-uuid-string"
}
```

### `/ask` Response (streamed NDJSON)
```json
{"type": "token",   "data": "The Bhagavad Gita teaches..."}
{"type": "token",   "data": " compassion as..."}
{"type": "sources", "data": [{"book": "Bhagavad Gita 2:47", "page": "2:47", "snippet": "..."}]}
```
Cache hits return a single `{"type": "cache", "data": {"answer": "...", "sources": [...]}}` line.

---

## 📝 Notes

- The LLM is instructed **never** to answer from outside the provided texts
- Each response includes **source citations** (book + chapter/verse where available)
- Responses synthesize wisdom **across all books** when relevant
- The semantic cache skips the LLM for repeated or near-identical questions (cosine distance < 0.35)
- Follow-up retrieval automatically augments vague short queries with the previous question for better semantic matching

---

## 🗺️ Planned Features

- Contextual chunk expansion (fetch ±1 surrounding chunks)
- HyDE — Hypothetical Document Embedding for abstract queries
- Answer faithfulness scoring (LLM-as-judge)
- Query rewriting for vague inputs
- Snippet preview on source hover
- Query suggestions after each answer
- Compare mode — side-by-side view across books
- Hallucination guardrail
- Out-of-scope detection
- Rate limiting & API key hardening

---

## 🎬 Demo

App Link: https://shouvik99-lifeguide.hf.space/