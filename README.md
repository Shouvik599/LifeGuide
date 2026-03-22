---
title: Sacred Texts RAG
emoji: 🕊️
colorFrom: gold
colorTo: white
sdk: docker
app_port: 7860
pinned: false
---

# 🕊️ Sacred Texts RAG — Multi-Religion Knowledge Base

A Retrieval-Augmented Generation (RAG) application that answers spiritual queries using Bhagavad Gita, Quran, Bible and the Guru Granth Sahib as the sole knowledge sources.

---

## 📁 Project Structure

```
sacred-texts-rag/
├── README.md
├── requirements.txt
├── .env.example
├── ingest.py               # Step 1: Load PDFs → chunk → embed → store
├── rag_chain.py            # Core RAG chain logic
├── app.py                  # FastAPI backend server
└── frontend/
    └── index.html          # Chat UI (open in browser)
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
# Edit .env and add your GEMINI_API_KEY
```

### 3. Add Your PDF Books
Place your PDF files in a `books/` folder:
```
books/
├── bhagavad_gita.pdf
├── quran.pdf
└── bible.pdf
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
Server runs at: `http://localhost:8000`

### 6. Open the Frontend
Open `frontend/index.html` in your browser — no server needed for the UI.

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Your Google Gemini API key |
| `NVIDIA_API_KEY` | Your NVIDIA API key |
| `CHROMA_DB_PATH` | Path to ChromaDB storage (default: `./chroma_db`) |
| `CHUNKS_PER_BOOK` | Number of chunks to retrieve per query (default: `3`) |

---

## 🧠 How It Works

```
User Query
    │
    ▼
[Embedding Model]  ←── NVIDIA llama-nemotron-embed-vl-1b-v2
    │
    ▼
[ChromaDB Vector Store]  ←── Semantic similarity search
    │  (retrieves top-K chunks from Gita, Quran, Bible, and the Guru Granth Sahib)
    │
    ▼
[Prompt with Context]
    │
    ▼
[Gemini 2.5 Flash Lite]  ←── Answer grounded ONLY in retrieved texts
    │
    ▼
Response with source citations (book + chapter/verse)
```

---

## 📝 Notes

- The LLM is instructed **never** to answer from outside the provided texts
- Each response includes **source citations** (which book the answer came from)
- Responses synthesize wisdom **across all books** when relevant
