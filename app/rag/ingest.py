"""
Ingests policy documents from data/policy_docs/ into the vector store.

Usage:
    python -m app.rag.ingest
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

POLICY_DOCS_DIR = Path("data/policy_docs")
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 100
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pgvector")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def load_documents() -> list[tuple[str, str]]:
    """Returns list of (source_filename, full_text)."""
    docs = []
    for path in POLICY_DOCS_DIR.glob("*.txt"):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    return docs


def ingest_chroma():
    import chromadb

    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
    collection = client.get_or_create_collection("policy_docs")

    ids, texts, metadatas = [], [], []
    for source, text in load_documents():
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{source}::{i}")
            texts.append(chunk)
            metadatas.append({"source": source})

    if not ids:
        print("No documents found in data/policy_docs/. Add .txt files first.")
        return

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(ids)} chunks from {len(set(m['source'] for m in metadatas))} documents into Chroma.")


def ingest_pgvector():
    from app.rag.retriever import _embed
    from app.services.db_pool import pooled_conn

    with pooled_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_chunks (
                id SERIAL PRIMARY KEY,
                source TEXT,
                chunk_text TEXT,
                embedding VECTOR(384)
            )
            """
        )
        # Clear existing rows first so re-running ingestion is idempotent —
        # without this, every re-run (e.g. after adding a new policy doc)
        # would stack duplicate copies of previously-ingested chunks on top
        # of the new ones, since this is a plain INSERT with no dedup key.
        cur.execute("DELETE FROM policy_chunks")

        count = 0
        for source, text in load_documents():
            for chunk in chunk_text(text):
                embedding = _embed(chunk)
                cur.execute(
                    "INSERT INTO policy_chunks (source, chunk_text, embedding) VALUES (%s, %s, %s)",
                    (source, chunk, embedding),
                )
                count += 1
        conn.commit()
        cur.close()
    print(f"Ingested {count} chunks into pgVector (existing rows cleared first).")


if __name__ == "__main__":
    if VECTOR_BACKEND == "chroma":
        ingest_chroma()
    elif VECTOR_BACKEND == "pgvector":
        ingest_pgvector()
    else:
        raise NotImplementedError(f"Unsupported VECTOR_BACKEND: {VECTOR_BACKEND}")
