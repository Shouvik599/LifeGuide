"""
rag_chain.py — Core RAG chain using LangChain + Gemini.

KEY FIX: Uses per-book retrieval (guaranteed slots per scripture) instead of
a single similarity search — so no book gets starved from the context window
when the query is semantically closer to another book's language.

This module exposes a single function:
    answer = query_sacred_texts(user_question)

Returns a dict with:
    {
        "answer": "...",
        "sources": [
            {"book": "Bhagavad Gita", "page": 42, "snippet": "..."},
            ...
        ]
    }
"""

import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sacred_texts")

# Chunks retrieved PER BOOK — guarantees every scripture contributes to the answer
CHUNKS_PER_BOOK = int(os.getenv("CHUNKS_PER_BOOK", "3"))

# All books currently in the knowledge base — add new books here as you ingest them
KNOWN_BOOKS = [
    "Bhagavad Gita",
    "Quran",
    "Bible",
    "Guru Granth Sahib",
]


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scholarly and compassionate guide to sacred scriptures.
You have deep knowledge of the Bhagavad Gita, the Quran, the Bible, and the Guru Granth Sahib.

STRICT RULES you must ALWAYS follow:
1. Answer ONLY using the provided context passages. Do NOT use any external knowledge.
2. If a specific book's passages are provided but not relevant to the question, skip that book.
3. If NONE of the context is relevant, say: "The provided texts do not directly address this question."
4. Always cite which book(s) your answer draws from.
5. When the question asks to COMPARE books (e.g. "what do Quran and Gita say"), you MUST
   address EACH of those books separately, then synthesise the common thread.
6. Be respectful and neutral toward all faiths — treat each text with equal reverence.
7. Do NOT speculate, invent verses, or add information beyond the context.

FORMAT your response as:
- A clear, thoughtful answer (2–4 paragraphs)
- A "📚 Sources" section listing each book referenced with the key insight drawn from it

Context passages from the sacred texts (guaranteed passages from each book):
────────────────────────────────────────
{context}
────────────────────────────────────────
"""

HUMAN_PROMPT = "Question: {question}"


# ─── Embeddings & Vector Store ────────────────────────────────────────────────

def get_embeddings():
    return NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-vl-1b-v2",
        api_key=NVIDIA_API_KEY,
        truncate="NONE",
    )


def get_vector_store(embeddings):
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


# ─── Per-Book Retrieval ───────────────────────────────────────────────────────

def retrieve_per_book(question: str, vector_store: Chroma) -> list:
    """
    Retrieve CHUNKS_PER_BOOK chunks from EACH known book independently,
    using a metadata filter. This guarantees every scripture is represented
    in the context — no book can be crowded out by higher-scoring chunks
    from another book.
    """
    all_docs = []
    for book in KNOWN_BOOKS:
        try:
            results = vector_store.similarity_search(
                query=question,
                k=CHUNKS_PER_BOOK,
                filter={"book": book},          # ← metadata filter: only this book
            )
            if results:
                print(f"  📖  {book}: {len(results)} chunk(s) retrieved")
            else:
                print(f"  ⚠️   {book}: 0 chunks found (not ingested?)")
            all_docs.extend(results)
        except Exception as e:
            print(f"  ❌  {book}: retrieval error — {e}")

    return all_docs


# ─── Format Retrieved Docs ────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    """
    Format retrieved documents grouped by book for clarity.
    Each chunk is labelled with book and page number.
    """
    # Group by book to keep context readable
    by_book: dict[str, list] = {}
    for doc in docs:
        book = doc.metadata.get("book", "Unknown")
        by_book.setdefault(book, []).append(doc)

    sections = []
    for book, book_docs in by_book.items():
        header = f"═══ {book} ═══"
        chunks = []
        for i, doc in enumerate(book_docs, 1):
            page = doc.metadata.get("page", "?")
            chunks.append(f"  [{i}] (Page {page}): {doc.page_content.strip()}")
        sections.append(header + "\n" + "\n\n".join(chunks))

    return "\n\n".join(sections)


# ─── Build the RAG Chain ──────────────────────────────────────────────────────

def build_chain():
    """Build and return the LLM chain and vector store."""
    embeddings = get_embeddings()
    vector_store = get_vector_store(embeddings)

    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=NVIDIA_API_KEY,
        temperature=0.2,
        top_p=0.7,
        max_output_tokens=2048,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    # Chain: prompt → LLM → string output
    # (retrieval is handled manually in query_sacred_texts for per-book control)
    llm_chain = prompt | llm | StrOutputParser()

    return llm_chain, vector_store


# ─── Public API ───────────────────────────────────────────────────────────────

_llm_chain = None
_vector_store = None


def query_sacred_texts(question: str) -> dict:
    """
    Query the sacred texts knowledge base with guaranteed per-book retrieval.

    Args:
        question: The user's spiritual/philosophical question.

    Returns:
        {
            "answer": str,
            "sources": list[dict]   # [{book, page, snippet}, ...]
        }
    """
    global _llm_chain, _vector_store

    if _llm_chain is None:
        print("🔧  Initialising RAG chain (first call)...")
        _llm_chain, _vector_store = build_chain()

    # Step 1: Retrieve per-book (guaranteed slots for every scripture)
    print(f"\n🔍  Retrieving {CHUNKS_PER_BOOK} chunks per book for: '{question}'")
    source_docs = retrieve_per_book(question, _vector_store)

    if not source_docs:
        return {
            "answer": "No content found in the knowledge base. Please run ingest.py first.",
            "sources": [],
        }

    # Step 2: Format context grouped by book
    context = format_docs(source_docs)

    # Step 3: Generate answer
    answer = _llm_chain.invoke({"context": context, "question": question})

    # Step 4: Build deduplicated source list for the UI
    seen_books = set()
    sources = []
    for doc in source_docs:
        book = doc.metadata.get("book", "Unknown")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:200].strip() + "..."
        if book not in seen_books:
            seen_books.add(book)
            sources.append({"book": book, "page": page, "snippet": snippet})

    return {
        "answer": answer,
        "sources": sources,
    }


# ─── Quick CLI Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_q = "In what aspects do the Quran and Gita teach the same thing?"
    print(f"\n🔍  Test query: {test_q}\n")
    result = query_sacred_texts(test_q)
    print("📝  Answer:\n")
    print(result["answer"])
    print("\n📚  Sources retrieved:")
    for s in result["sources"]:
        print(f"  - {s['book']} (page {s['page']})")