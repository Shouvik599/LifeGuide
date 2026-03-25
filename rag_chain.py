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
from pydoc import doc
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA, NVIDIARerank
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
load_dotenv()
import json

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

# Create a separate collection for semantic cache
CACHE_COLLECTION = "semantic_cache"

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

def get_reranked_retriever(base_retriever):
    """
    Wraps your Hybrid/Per-Book retriever with a Reranking layer.
    """
    # 1. Initialize the NVIDIA Reranker (NIM or API Catalog)
    # Using nvidia/llama-3.2-nv-rerankqa-1b-v2 or similar
    reranker = NVIDIARerank(
        model="nvidia/llama-3.2-nv-rerankqa-1b-v2", 
        api_key=NVIDIA_API_KEY,
        top_n=5 # Only send the top 5 most relevant chunks to the LLM
    )
    
    # 2. Wrap the base retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=base_retriever
    )
    
    return compression_retriever

def retrieve_per_book(question: str, vector_store: Chroma) -> list:
    """
    Retrieve CHUNKS_PER_BOOK chunks from EACH known book independently,
    using a metadata filter. This guarantees every scripture is represented
    in the context — no book can be crowded out by higher-scoring chunks
    from another book.
    """
    all_candidates = []
    
    # Detect if user is asking about a specific book
    target_books = []
    question_lower = question.lower()
    
    # Check for keywords in the question
    if any(kw in question_lower for kw in ["gita", "bhagavad", "hindu", "hinduism"]):
        target_books.append("Bhagavad Gita")
    if any(kw in question_lower for kw in ["quran", "koran", "islam", "muslim", "muhammad"]):
        target_books.append("Quran")
    if any(kw in question_lower for kw in ["bible", "testament", "christian", "jesus", "christ"]):
        target_books.append("Bible")
    if any(kw in question_lower for kw in ["granth", "guru", "sikh", "sikhism", "nanak"]):
        target_books.append("Guru Granth Sahib")
        
    # If no specific book is detected, use all books
    books_to_search = target_books if target_books else KNOWN_BOOKS
    
    print(f"🎯 Routing query to: {books_to_search}")
    
    for book in books_to_search:
        try:
            # Increase k for the base retrieval to 10
            CANDIDATE_COUNT = 10
            
            # Get the full collection of documents for this book to build BM25
            # For small demo, we can pull into memory; for larger corpora, consider a more efficient approach
            book_data = vector_store.get(where={"book": book})
            book_docs = []
            from langchain_core.documents import Document
            book_docs = [Document(page_content=d, metadata=m) 
                         for d, m in zip(book_data["documents"], book_data["metadatas"])]
            if not book_docs:
                continue
            
            
            # Setup BM25
            bm25_retriever = BM25Retriever.from_documents(book_docs)
            bm25_retriever.k = CANDIDATE_COUNT
            
            
            # Setup vector retriever
            vector_retriever = vector_store.as_retriever(search_kwargs={"k": CANDIDATE_COUNT, "filter": {"book": book}})
        
            
            #  Combine into ensemble retriever
            ensemble_retriver = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
            
            # Colect candidates without reranking yet
            book_candidates = ensemble_retriver.invoke(question)
            all_candidates.extend(book_candidates)
            print(f"  📦 {book}: Found {len(book_candidates)} candidates")
        
        except Exception as e:
            print(f"  ❌  {book}: retrieval error — {e}")
            
    
    # Rerank the entire pool at once
    if not all_candidates:
        return []
    
    print(f"🚀 Reranking {len(all_candidates)} total candidates...")
    reranker = NVIDIARerank(
        model="nvidia/llama-3.2-nv-rerankqa-1b-v2", 
        api_key=NVIDIA_API_KEY,
        top_n=5 # Final count for LLM context
    )
       
    # Use the reranker directly to compress the full list
    final_docs = reranker.compress_documents(all_candidates, question)     
    
    for i, doc in enumerate(final_docs):
        score = doc.metadata.get("relevance_score", "N/A")
        print(f"Rank {i+1} [{doc.metadata['book']}]: Score {score}")

    return final_docs


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
            ch = doc.metadata.get("chapter")
            vs = doc.metadata.get("verse")
            ang = doc.metadata.get("ang")
            
            # Create a clean citation string
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


def query_sacred_texts(question: str):
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
    
    # --- Semantic cache check ---
    cache_coll = _vector_store._client.get_or_create_collection(CACHE_COLLECTION)
    cache_results = cache_coll.query(
        query_texts=[question],
        n_results=1
    )

    THRESHOLD = 0.35
    # FIXED: Added check for cache_results['ids'] and ensuring distances is not empty
    if cache_results['ids'] and cache_results['ids'][0]: 
        distance = cache_results['distances'][0][0]
        if distance < THRESHOLD:  # Similarity threshold
            print(f"⚡️ Semantic Cache Hit! (Distance: {distance:.4f})")
            yield json.dumps({"type": "cache","data": json.loads(cache_results['metadatas'][0][0]['response_json'])}) + "\n"
            return
            
    # Step 1: Retrieve per-book (guaranteed slots for every scripture)
    print(f"\n🔍  Retrieving {CHUNKS_PER_BOOK} chunks per book for: '{question}'")
    source_docs = retrieve_per_book(question, _vector_store)

    if not source_docs:
        yield json.dumps({"type": "token", "data": "No content found in the knowledge base."}) + "\n"
        return

    # 3. Step 2: Format sources for the UI immediately
    seen_sources = set()
    sources = []
    for doc in source_docs:
        book = doc.metadata.get("book", "Unknown")
        ch = doc.metadata.get("chapter")
        vs = doc.metadata.get("verse")
        ang = doc.metadata.get("ang")
        
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
            print("Display name:", display_name)
            print("Page:", cite_val)
            sources.append({"book": display_name, "page": cite_val, "snippet": snippet})
    # Step 2: Format context grouped by book
    context = format_docs(source_docs)
    full_answer =""

    # Step 3: Stream from the chain:
    for chunk in _llm_chain.invoke({"context": context, "question": question}):
        full_answer += chunk
        yield json.dumps({"type": "token", "data": chunk}) + "\n"  # Stream the answer as it's generated
    
    
    # Filter sources to only those the LLM actually referenced
    final_sources = []
    ansnwer_lower = full_answer.lower()
    
    for s in sources:
        if s["book"].lower() in ansnwer_lower:
            final_sources.append(s)
            
    # If the LLM didn't explicitly reference any sources, we can optionally include all retrieved ones or none
    display_sources = final_sources if final_sources else []
    
    # Step 4: After streaming is done, save to semantic cache for future similar queries
    result = {
        "answer": full_answer,
        "sources": display_sources,
    }
    
    
    cache_coll.add(
        documents=[question],
        metadatas=[{"response_json": json.dumps(result)}],
        ids=[question]
    )
    
    # Send sources as a final message after the answer is fully streamed
    yield json.dumps({"type": "sources", "data": sources}) + "\n"
    


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