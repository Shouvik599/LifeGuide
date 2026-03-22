"""
ingest.py — Step 1: Build the vector knowledge base from religious PDFs.

Run this ONCE before starting the app:
    python ingest.py

It will:
1. Load all PDFs from the ./books/ directory
2. Split them into overlapping semantic chunks
3. Embed each chunk using NVIDIA's llama-nemotron embedding model
4. Persist everything into a local ChromaDB vector store
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────

BOOKS_DIR = Path("./books")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sacred_texts")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Mapping of filename keywords → friendly book name stored in metadata
BOOK_NAME_MAP = {
    "gita": "Bhagavad Gita",
    "bhagavad": "Bhagavad Gita",
    "quran": "Quran",
    "koran": "Quran",
    "bible": "Bible",
    "testament": "Bible",
    "granth": "Guru Granth Sahib",    # ← ADD
    "guru": "Guru Granth Sahib",      # ← ADD
}
# Chunk settings — tuned for religious texts (verses are short)
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap to preserve verse context across boundaries


# ─── Helpers ──────────────────────────────────────────────────────────────────

def detect_book_name(filename: str) -> str:
    """Infer the book's display name from its filename."""
    name_lower = filename.lower()
    for keyword, book_name in BOOK_NAME_MAP.items():
        if keyword in name_lower:
            return book_name
    # Fallback: use the filename stem, title-cased
    return Path(filename).stem.replace("_", " ").title()


def load_pdf(pdf_path: Path) -> list:
    """
    Load a PDF using PyMuPDF (preferred) or PyPDF as fallback.
    Returns a list of LangChain Document objects.
    """
    try:
        loader = PyMuPDFLoader(str(pdf_path))
        print(f"  📖 Loading with PyMuPDF: {pdf_path.name}")
    except Exception:
        loader = PyPDFLoader(str(pdf_path))
        print(f"  📖 Loading with PyPDF: {pdf_path.name}")

    docs = loader.load()
    print(f"     → {len(docs)} pages loaded")
    return docs


def tag_documents(docs: list, book_name: str, source_file: str) -> list:
    """
    Enrich each document's metadata with:
    - book: display name (e.g. "Bhagavad Gita")
    - source_file: original filename
    """
    for doc in docs:
        doc.metadata["book"] = book_name
        doc.metadata["source_file"] = source_file
        # Keep the page number if already present from the loader
        if "page" not in doc.metadata:
            doc.metadata["page"] = 0
    return docs


# ─── Main Ingestion ───────────────────────────────────────────────────────────

def ingest():
    if not NVIDIA_API_KEY:
        print("❌  NVIDIA_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    if not BOOKS_DIR.exists():
        print(f"❌  Books directory not found: {BOOKS_DIR.resolve()}")
        print("    Create a ./books/ folder and add your PDFs there.")
        sys.exit(1)

    pdf_files = list(BOOKS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"❌  No PDF files found in {BOOKS_DIR.resolve()}")
        sys.exit(1)

    print(f"\n🕊️  Sacred Texts RAG — Ingestion Pipeline")
    print(f"{'─' * 50}")
    print(f"📂  Books directory : {BOOKS_DIR.resolve()}")
    print(f"💾  ChromaDB path   : {Path(CHROMA_DB_PATH).resolve()}")
    print(f"📚  PDFs found      : {len(pdf_files)}")
    print(f"{'─' * 50}\n")

    # ── Step 1: Load all PDFs ────────────────────────────────────────────────
    all_docs = []
    for pdf_path in pdf_files:
        book_name = detect_book_name(pdf_path.name)
        print(f"📕  {book_name}")
        raw_docs = load_pdf(pdf_path)
        tagged_docs = tag_documents(raw_docs, book_name, pdf_path.name)
        all_docs.extend(tagged_docs)
        print(f"     ✅  Tagged as '{book_name}'\n")

    print(f"📄  Total pages loaded: {len(all_docs)}")

    # ── Step 2: Split into chunks ────────────────────────────────────────────
    print(f"\n✂️   Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Respect paragraph/verse boundaries
    )
    chunks = splitter.split_documents(all_docs)
    print(f"     → {len(chunks)} chunks created")

    # ── Step 3: Embed & store ────────────────────────────────────────────────
    print(f"\n🔢  Initialising NVIDIA embedding model (llama-nemotron-embed-vl-1b-v2)...")
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-nemotron-embed-vl-1b-v2",
        api_key=NVIDIA_API_KEY,
        truncate="NONE",
    )

    print(f"💾  Building ChromaDB vector store — this may take a few minutes...")
    print(f"     (Embedding {len(chunks)} chunks...)\n")

    # Process in batches to avoid rate limits
    BATCH_SIZE = 100
    vector_store = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches}: embedding {len(batch)} chunks...")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_name=COLLECTION_NAME,
            )
        else:
            vector_store.add_documents(batch)

    print(f"\n{'─' * 50}")
    print(f"✅  Ingestion complete!")
    print(f"    📦  {len(chunks)} chunks stored in ChromaDB")
    print(f"    📂  Location: {Path(CHROMA_DB_PATH).resolve()}")
    print(f"\n👉  Now run: python app.py")
    print(f"{'─' * 50}\n")


if __name__ == "__main__":
    ingest()