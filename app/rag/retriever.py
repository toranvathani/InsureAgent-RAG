"""
Vector retrieval over ingested P&C policy documents.
Supports pgVector (Postgres) or Chroma, chosen via VECTOR_BACKEND env var.
Run app/rag/ingest.py first to populate the store from data/policy_docs/.
"""
import os
from typing import List, TypedDict

from dotenv import load_dotenv

load_dotenv()

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "pgvector")  # chroma | pgvector

_embedder = None  # lazy-loaded singleton, avoids reloading the model on every call


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        # Runs locally, no API key needed. 384-dim output, good quality for this scale.
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


class RetrievedChunk(TypedDict):
    source: str
    text: str
    score: float


def _embed(text: str) -> list[float]:
    model = _get_embedder()
    return model.encode(text, normalize_embeddings=True).tolist()


def retrieve_policy_context(query: str, top_k: int = 5) -> List[RetrievedChunk]:
    if not query.strip():
        return []

    if VECTOR_BACKEND == "chroma":
        return _retrieve_chroma(query, top_k)
    elif VECTOR_BACKEND == "pgvector":
        return _retrieve_pgvector(query, top_k)
    else:
        raise NotImplementedError(f"Unsupported VECTOR_BACKEND: {VECTOR_BACKEND}")


def _retrieve_chroma(query: str, top_k: int) -> List[RetrievedChunk]:
    import chromadb

    client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
    collection = client.get_or_create_collection("policy_docs")

    results = collection.query(query_texts=[query], n_results=top_k)
    chunks: List[RetrievedChunk] = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"source": meta.get("source", "unknown"), "text": doc, "score": 1 - dist})
    return chunks


def _retrieve_pgvector(query: str, top_k: int) -> List[RetrievedChunk]:
    from app.services.db_pool import pooled_conn

    query_embedding = _embed(query)
    with pooled_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT source, chunk_text, 1 - (embedding <=> %s::vector) AS score
            FROM policy_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        )
        rows = cur.fetchall()
        cur.close()
    return [{"source": r[0], "text": r[1], "score": r[2]} for r in rows]
