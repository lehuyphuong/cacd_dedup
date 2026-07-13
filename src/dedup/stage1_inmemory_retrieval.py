"""
CACD Stage 1 — In-Memory Coarse Retrieval.

Role: narrow the candidate set from the full index down to K nearest
neighbours before Stage 2's cross-encoder scores them.

The index is a single growing in-memory matrix of unit-normalized chunk
embeddings, searched with an exact (not approximate) vectorized
matrix-vector product. This finds the true top-K nearest neighbours under
cosine similarity, unlike an approximate index such as HNSW, at O(pool_size)
cost per query instead of O(log pool_size). At the corpus sizes this
project evaluates (roughly 2,000-16,000 chunks per configuration), the
per-query cost of an in-process matrix multiply is small enough that this
trade-off is favorable in practice.

Qdrant is not used during this stage; it is only used once, at the end of
a full ingest run, to persist the final kept chunks for retrieval
evaluation (see src/dedup/stage3_decision.py and src/ingestion/vector_store.py).

Public API:
  InMemoryIndex(dim)    -- one instance per chunking-strategy config.
  .top_k(query_vecs, k) -- batched top-K search; returns candidate dicts
                           with the same shape as a Qdrant hit
                           (chunk_id, doc_id, title, text, score,
                           parent_id, level).
  .add(chunks, vecs)    -- append newly-kept chunks to the pool.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize rows so a dot product equals cosine similarity.
    Defensive: the embedder is expected to already output unit vectors,
    but correctness here should not silently depend on that."""
    if vecs.size == 0:
        return vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return vecs / norms


class InMemoryIndex:
    """
    Growing in-memory pool of (vector, metadata) for chunks already kept
    in the current config's ingest run. Create one fresh instance per
    chunking configuration.
    """

    def __init__(self, dim: int):
        self._dim = dim
        self._vecs = np.empty((0, dim), dtype=np.float32)
        self._meta: list[dict] = []

    def __len__(self) -> int:
        return len(self._meta)

    def top_k(self, query_vecs: list[list[float]], k: int) -> list[list[dict]]:
        """
        Batched top-K cosine search against the current pool.

        Args:
            query_vecs: embeddings for the new chunks in this micro-batch,
                        shape (m, dim).
            k         : number of candidates to return per query.

        Returns:
            List of length m; each entry is a list of up to k candidate
            dicts (chunk_id, doc_id, title, text, score, parent_id, level),
            sorted by descending cosine score. Empty list for a query if
            the pool is currently empty.
        """
        m = len(query_vecs)
        if m == 0:
            return []
        if len(self._meta) == 0:
            return [[] for _ in range(m)]

        q = _l2_normalize(np.asarray(query_vecs, dtype=np.float32))
        sims = q @ self._vecs.T  # (m, pool_size) -- one BLAS matmul, no I/O

        pool_size = self._vecs.shape[0]
        k_eff = min(k, pool_size)

        results: list[list[dict]] = []
        # argpartition finds the top-k unordered in O(pool_size), then we
        # sort just those k -- avoids a full O(pool_size log pool_size)
        # sort per query when k << pool_size.
        for row in range(m):
            row_sims = sims[row]
            if k_eff < pool_size:
                idx = np.argpartition(-row_sims, k_eff - 1)[:k_eff]
            else:
                idx = np.arange(pool_size)
            idx = idx[np.argsort(-row_sims[idx])]

            cands = []
            for j in idx:
                meta = self._meta[int(j)]
                cands.append({
                    "chunk_id":  meta["chunk_id"],
                    "doc_id":    meta["doc_id"],
                    "title":     meta["title"],
                    "text":      meta["text"],
                    "score":     round(float(row_sims[j]), 4),
                    "parent_id": meta.get("parent_id"),
                    "level":     meta.get("level"),
                })
            results.append(cands)
        return results

    def add(self, chunks: list[dict], vecs: list[list[float]]) -> None:
        """
        Append newly-kept chunks to the pool. Call once per micro-batch,
        after Stage 3 has decided which chunks in that batch to keep.
        Chunks within the same micro-batch never see each other; only
        chunks added by earlier batches are visible to later ones.
        """
        if not chunks:
            return
        new_vecs = _l2_normalize(np.asarray(vecs, dtype=np.float32))
        self._vecs = (
            np.vstack([self._vecs, new_vecs]) if self._vecs.shape[0] else new_vecs
        )
        for c in chunks:
            self._meta.append({
                "chunk_id":  c["chunk_id"],
                "doc_id":    c["doc_id"],
                "title":     c.get("title", ""),
                "text":      c["text"],
                "parent_id": c.get("parent_id"),
                "level":     c.get("level"),
            })
