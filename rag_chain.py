"""
rag_chain.py — Core RAG chain using LangChain + NVIDIA.

KEY FEATURES:
- Per-book retrieval (guaranteed slots per scripture)
- Hybrid BM25 + vector search with NVIDIA reranking
- Semantic cache for repeated/similar questions
- Multi-turn conversation memory (session-based ConversationBufferMemory)

Public API:
    query_sacred_texts(question, session_id) -> Generator[str, None, None]
    clear_session(session_id)
"""

import os
import json
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA, NVIDIARerank
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_core.documents import Document

load_dotenv()

NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY")
CHROMA_DB_PATH    = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME", "sacred_texts")
CHUNKS_PER_BOOK   = int(os.getenv("CHUNKS_PER_BOOK", "3"))
CACHE_COLLECTION  = "semantic_cache"
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))   # last N human+AI pairs kept

KNOWN_BOOKS = [
    "Bhagavad Gita",
    "Quran",
    "Bible",
    "Guru Granth Sahib",
]

# ─── In-memory session store ──────────────────────────────────────────────────
# { session_id: [HumanMessage | AIMessage, ...] }
_session_store: dict[str, list] = {}


def get_history(session_id: str) -> list:
    return _session_store.get(session_id, [])


def append_turn(session_id: str, human_msg: str, ai_msg: str):
    history = _session_store.setdefault(session_id, [])
    history.append(HumanMessage(content=human_msg))
    history.append(AIMessage(content=ai_msg))
    # Trim to last MAX_HISTORY_TURNS pairs (each pair = 2 messages)
    if len(history) > MAX_HISTORY_TURNS * 2:
        _session_store[session_id] = history[-(MAX_HISTORY_TURNS * 2):]


def clear_session(session_id: str):
    """Wipe the conversation history for a session."""
    _session_store.pop(session_id, None)


def list_sessions() -> list[str]:
    return list(_session_store.keys())


# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scholarly and compassionate guide to sacred scriptures.
You have deep knowledge of the Bhagavad Gita, the Quran, the Bible, and the Guru Granth Sahib.

STRICT RULES you must ALWAYS follow:
1. Answer ONLY using the provided context passages. Do NOT use any external knowledge.
2. If a specific book's passages are provided but not relevant to the question, skip that book.
3. If NONE of the context is relevant, say: "The provided texts do not directly address this question."
4. Always explicitly name and cite which book(s) your answer draws from in the text of your answer.
5. When the question asks to COMPARE books (e.g. "what do Quran and Gita say"), you MUST
   address EACH of those books separately, then synthesise the common thread.
6. Be respectful and neutral toward all faiths — treat each text with equal reverence.
7. Do NOT speculate, invent verses, or add information beyond the context.
8. You have access to the conversation history. Use it to:
   - Understand follow-up questions (e.g. "elaborate on the second point", "what about the Bible?")
   - Maintain continuity across turns without repeating yourself unnecessarily
   - Resolve pronouns and references ("it", "that teaching", "the verse you mentioned") from history

FORMAT your response as:
- A clear, thoughtful answer (2–4 paragraphs)
- A "📚 Sources" section listing each book referenced with the key insight drawn from it

Context passages from the sacred texts (guaranteed passages from each book):
────────────────────────────────────────
{context}
────────────────────────────────────────
"""


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


# ─── Per-Book Hybrid Retrieval ────────────────────────────────────────────────

def retrieve_per_book(question: str, vector_store: Chroma) -> list:
    """
    Retrieve CHUNKS_PER_BOOK chunks from EACH known book independently using
    a hybrid BM25+vector ensemble, then rerank the pooled candidates.
    """
    all_candidates = []
    question_lower = question.lower()

    target_books = []
    if any(kw in question_lower for kw in ["gita", "bhagavad", "hindu", "hinduism"]):
        target_books.append("Bhagavad Gita")
    if any(kw in question_lower for kw in ["quran", "koran", "islam", "muslim", "muhammad"]):
        target_books.append("Quran")
    if any(kw in question_lower for kw in ["bible", "testament", "christian", "jesus", "christ"]):
        target_books.append("Bible")
    if any(kw in question_lower for kw in ["granth", "guru", "sikh", "sikhism", "nanak"]):
        target_books.append("Guru Granth Sahib")

    books_to_search = target_books if target_books else KNOWN_BOOKS
    print(f"🎯 Routing query to: {books_to_search}")

    CANDIDATE_COUNT = 10

    for book in books_to_search:
        try:
            book_data = vector_store.get(where={"book": book})
            book_docs = [
                Document(page_content=d, metadata=m)
                for d, m in zip(book_data["documents"], book_data["metadatas"])
            ]
            if not book_docs:
                continue

            bm25_retriever = BM25Retriever.from_documents(book_docs)
            bm25_retriever.k = CANDIDATE_COUNT

            vector_retriever = vector_store.as_retriever(
                search_kwargs={"k": CANDIDATE_COUNT, "filter": {"book": book}}
            )

            ensemble = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5],
            )

            book_candidates = ensemble.invoke(question)
            all_candidates.extend(book_candidates)
            print(f"  📦 {book}: {len(book_candidates)} candidates")

        except Exception as e:
            print(f"  ❌  {book}: retrieval error — {e}")

    if not all_candidates:
        return []

    print(f"🚀 Reranking {len(all_candidates)} total candidates...")
    reranker = NVIDIARerank(
        model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        api_key=NVIDIA_API_KEY,
        top_n=5,
    )
    final_docs = reranker.compress_documents(all_candidates, question)

    for i, doc in enumerate(final_docs):
        score = doc.metadata.get("relevance_score", "N/A")
        print(f"Rank {i+1} [{doc.metadata['book']}]: Score {score}")

    return final_docs


# ─── Format Retrieved Docs ────────────────────────────────────────────────────

def format_docs(docs: list) -> str:
    by_book: dict[str, list] = {}
    for doc in docs:
        book = doc.metadata.get("book", "Unknown")
        by_book.setdefault(book, []).append(doc)

    sections = []
    for book, book_docs in by_book.items():
        header = f"═══ {book} ═══"
        chunks = []
        for i, doc in enumerate(book_docs, 1):
            ang = doc.metadata.get("ang")
            ch  = doc.metadata.get("chapter")
            vs  = doc.metadata.get("verse")
            if ang:
                citation = f"Ang {ang}"
            elif ch and vs:
                citation = f"{ch}:{vs}"
            else:
                citation = f"Page {doc.metadata.get('page', '?')}"
            chunks.append(f"  [{i}] ({citation}): {doc.page_content.strip()}")
        sections.append(header + "\n" + "\n\n".join(chunks))

    return "\n\n".join(sections)


# ─── Build the RAG Chain ──────────────────────────────────────────────────────

def build_chain():
    embeddings   = get_embeddings()
    vector_store = get_vector_store(embeddings)

    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        api_key=NVIDIA_API_KEY,
        temperature=0.2,
        top_p=0.7,
        max_output_tokens=2048,
    )

    # Prompt now includes a chat-history placeholder so prior turns are visible
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),   # ← injected per-request
        ("human", "{question}"),
    ])

    llm_chain = prompt | llm | StrOutputParser()
    return llm_chain, vector_store


# ─── Singleton init ───────────────────────────────────────────────────────────

_llm_chain    = None
_vector_store = None


# ─── Public API ───────────────────────────────────────────────────────────────

def query_sacred_texts(question: str, session_id: str = "default"):
    """
    Stream an answer grounded in the sacred texts, maintaining per-session
    conversation history for natural follow-up questions.

    Yields JSON-lines of the form:
        {"type": "token",   "data": "<chunk>"}
        {"type": "sources", "data": [...]}
        {"type": "cache",   "data": {"answer": "...", "sources": [...]}}
    """
    global _llm_chain, _vector_store

    if _llm_chain is None:
        print("🔧  Initialising RAG chain (first call)...")
        _llm_chain, _vector_store = build_chain()

    # ── Semantic cache check (skip for follow-ups that reference history) ──
    history = get_history(session_id)
    is_followup = len(history) > 0

    if not is_followup:
        cache_coll = _vector_store._client.get_or_create_collection(CACHE_COLLECTION)
        cache_results = cache_coll.query(query_texts=[question], n_results=1)

        THRESHOLD = 0.35
        if cache_results["ids"] and cache_results["ids"][0]:
            distance = cache_results["distances"][0][0]
            if distance < THRESHOLD:
                print(f"⚡️ Semantic Cache Hit! (Distance: {distance:.4f})")
                cached = json.loads(cache_results["metadatas"][0][0]["response_json"])
                # Store this cache hit in session memory too
                append_turn(session_id, question, cached["answer"])
                yield json.dumps({"type": "cache", "data": cached}) + "\n"
                return

    # ── Retrieval ──────────────────────────────────────────────────────────
    # For follow-ups, augment the question with the last human turn for better
    # semantic search (the follow-up itself may be too short/vague)
    retrieval_query = question
    if is_followup and len(question.split()) < 8:
        last_human = next(
            (m.content for m in reversed(history) if isinstance(m, HumanMessage)), ""
        )
        retrieval_query = f"{last_human} {question}".strip()
        print(f"🔁 Follow-up detected — augmented retrieval query: '{retrieval_query}'")

    print(f"\n🔍  Retrieving chunks for: '{retrieval_query}'")
    source_docs = retrieve_per_book(retrieval_query, _vector_store)

    if not source_docs:
        yield json.dumps({"type": "token", "data": "No content found in the knowledge base."}) + "\n"
        return

    # ── Build sources list ─────────────────────────────────────────────────
    seen_sources: set[str] = set()
    sources = []
    for doc in source_docs:
        book = doc.metadata.get("book", "Unknown")
        ang  = doc.metadata.get("ang")
        ch   = doc.metadata.get("chapter")
        vs   = doc.metadata.get("verse")
        if ang:
            cite_val = f"Ang {ang}"
        elif ch and vs:
            cite_val = f"{ch}:{vs}"
        else:
            cite_val = f"p. {doc.metadata.get('page', '?')}"
        display_name = f"{book} {cite_val}"
        snippet = doc.page_content[:200].strip() + "..."
        if display_name not in seen_sources:
            seen_sources.add(display_name)
            sources.append({"book": display_name, "page": cite_val, "snippet": snippet})

    context   = format_docs(source_docs)
    full_answer = ""

    # ── Stream LLM response (history injected here) ────────────────────────
    for chunk in _llm_chain.stream({
        "context":  context,
        "question": question,
        "history":  history,          # ← the conversation so far
    }):
        full_answer += chunk
        yield json.dumps({"type": "token", "data": chunk}) + "\n"

    # ── Filter sources to those actually cited in the answer ───────────────
    answer_lower = full_answer.lower()
    final_sources = [s for s in sources if s["book"].lower() in answer_lower] or []

    # ── Persist this turn into session memory ─────────────────────────────
    append_turn(session_id, question, full_answer)
    print(f"💾 Session '{session_id}': {len(get_history(session_id)) // 2} turn(s) stored")

    # ── Cache first-turn answers only ─────────────────────────────────────
    if not is_followup:
        result_to_cache = {"answer": full_answer, "sources": final_sources}
        try:
            cache_coll = _vector_store._client.get_or_create_collection(CACHE_COLLECTION)
            cache_coll.add(
                documents=[question],
                metadatas=[{"response_json": json.dumps(result_to_cache)}],
                ids=[question],
            )
        except Exception as e:
            print(f"⚠️  Cache write failed: {e}")

    yield json.dumps({"type": "sources", "data": sources}) + "\n"


# ─── Quick CLI Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_q = "What do the scriptures say about forgiveness?"
    print(f"\n🔍  Test query: {test_q}\n")
    for line in query_sacred_texts(test_q, session_id="cli-test"):
        obj = json.loads(line)
        if obj["type"] == "token":
            print(obj["data"], end="", flush=True)
    print("\n")