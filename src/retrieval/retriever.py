"""
Dense retrieval — cosine similarity via Qdrant.

Paper Section 3.4:
  "The experiments rely on dense retrieval in an embedding space. Each chunk
   and each query is encoded into a dense vector representation. Then retrieval
   is performed by nearest-neighbour search in a vector database."

No hybrid / BM25 — matching paper's evaluation protocol exactly.
"""

from __future__ import annotations

import logging
import time

from qdrant_client.models import SearchRequest

from configs.settings import TOP_K
from src.ingestion.embedder import embed_texts
from src.ingestion.vector_store import get_client

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    cname: str,
    top_k: int = TOP_K,
) -> tuple[list[dict], float]:
    """
    Retrieve top-k chunks for a query using dense cosine similarity.

    Returns (chunks, latency_ms).
    Each chunk dict contains payload fields + score.
    """
    client = get_client()
    t0 = time.perf_counter()

    # Embed query with the same model used for chunks
    dense_vec = embed_texts([query])[0]

    results = client.query_points(
        collection_name=cname,
        query=dense_vec,
        limit=top_k,
        with_payload=True,
    ).points

    latency_ms = (time.perf_counter() - t0) * 1000

    chunks = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append({
            "chunk_id":   payload.get("chunk_id", ""),
            "doc_id":     payload.get("doc_id", ""),
            "title":      payload.get("title", ""),
            "text":       payload.get("text", ""),
            "char_start": payload.get("char_start", 0),
            "char_end":   payload.get("char_end", 0),
            "score":      round(hit.score, 4),
        })

    return chunks, latency_ms
