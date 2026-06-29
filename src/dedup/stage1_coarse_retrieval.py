"""
CACD Stage 1 — Coarse Retrieval (bi-encoder + HNSW).

Role: narrow the candidate set from n (full index) down to K (small constant)
so that Stage 2 (cross-encoder, expensive) only processes K candidates
instead of all n chunks.

Algorithm:
  Batch-query Qdrant with all m vectors of the current document in a single
  request, retrieving the top-K nearest neighbours for each chunk.

Complexity: O(m log n) — same Big-O as plain HNSW.

Input:
  chunks     : list of chunk dicts (not yet inserted into Qdrant).
  dense_vecs : pre-computed embeddings for each chunk.
  cname      : Qdrant collection name (may already contain chunks from
               earlier iterations of the same ingest pass).
  top_k      : number of candidates to retrieve per chunk.

Output:
  (candidates_per_chunk, elapsed_seconds)
  candidates_per_chunk[i] = list of candidate dicts already in the index
                            that are nearest to chunks[i].
"""

from __future__ import annotations

import logging
import time

from qdrant_client.models import QueryRequest

from configs.settings import CACD_TOP_K_CANDIDATES
from src.ingestion.vector_store import get_client

logger = logging.getLogger(__name__)


def batch_coarse_retrieve(
    chunks: list[dict],
    dense_vecs: list[list[float]],
    cname: str,
    top_k: int | None = None,
) -> tuple[list[list[dict]], float]:
    """
    For each new chunk (embedding pre-computed), find the top-K nearest
    neighbours already present in the Qdrant collection (persistent index,
    not just the current batch).

    Args:
        chunks    : list of chunk dicts (not yet inserted into Qdrant).
        dense_vecs: embeddings corresponding to each chunk.
        cname     : Qdrant collection being ingested incrementally (may
                    already contain chunks from previous iterations).
        top_k     : K nearest neighbours per chunk.
                    None => use CACD_TOP_K_CANDIDATES from settings.

    Returns:
        (candidates_per_chunk, elapsed_seconds)
        candidates_per_chunk[i]: candidate dicts already in the index
                                 for chunks[i]; empty list if the
                                 collection has no points yet.
    """
    if top_k is None:
        top_k = CACD_TOP_K_CANDIDATES

    client = get_client()
    t0     = time.perf_counter()

    candidates_per_chunk: list[list[dict]] = []

    # Batch query — send all m vectors in a single request.
    # Qdrant processes each vector independently on the same fixed HNSW graph.
    try:
        requests = [
            QueryRequest(query=vec, limit=top_k, with_payload=True)
            for vec in dense_vecs
        ]
        results = client.query_batch_points(
            collection_name=cname,
            requests=requests,
        )
    except Exception as exc:
        # Empty collection or collection does not exist yet.
        logger.debug("Coarse retrieve: empty collection or error (%s)", exc)
        results = [None] * len(dense_vecs)

    for batch_result in results:
        if batch_result is None:
            candidates_per_chunk.append([])
            continue
        points = getattr(batch_result, "points", batch_result)
        cands  = []
        for hit in points:
            payload = hit.payload or {}
            cands.append({
                "chunk_id":  payload.get("chunk_id", ""),
                "doc_id":    payload.get("doc_id", ""),
                "title":     payload.get("title", ""),
                "text":      payload.get("text", ""),
                "score":     round(hit.score, 4),
                "parent_id": payload.get("parent_id"),
                "level":     payload.get("level"),
            })
        candidates_per_chunk.append(cands)

    elapsed            = time.perf_counter() - t0
    n_with_candidates  = sum(1 for c in candidates_per_chunk if c)
    logger.debug(
        "  Stage1 coarse retrieve: %d/%d chunks have candidates (top_k=%d) in %.2fs",
        n_with_candidates, len(chunks), top_k, elapsed,
    )
    return candidates_per_chunk, elapsed
