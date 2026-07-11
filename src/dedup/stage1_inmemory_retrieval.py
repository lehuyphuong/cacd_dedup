"""
CACD Stage 1 — In-Memory Coarse Retrieval (numpy top-K, no vector-store round trip).

Role: same as stage1_coarse_retrieval.py (narrow the candidate set from n down
to K before Stage 2's cross-encoder), but the "index I" from the paper is now
an in-memory, growing numpy array instead of a persistent Qdrant collection.

Why this exists (see conversation history / CACD_context_handoff.md):
  The original Qdrant-backed Stage 1 (stage1_coarse_retrieval.py) round-trips
  to QdrantClient(path=...) ("Local Mode") once per micro-batch. Local Mode
  is documented by qdrant-client itself as brute-force only (no HNSW) and
  SQLite-backed on disk -- "designed for development, testing, demos, and
  small-scale datasets (up to ~20,000 points)", not for an incremental
  upsert+query loop running hundreds of times per config. Measured on this
  project: per-chunk Stage 1 cost grew from ~6ms to ~60ms over a single
  9022-chunk run as the collection grew, dominating total ingest time (~74%)
  regardless of GPU speed, K, or Qdrant HNSW/optimizer settings (none of
  which Local Mode actually honours).

  rag_bench (the earlier, no-CACD baseline codebase for this paper) never
  hit this problem because its filters (Similarity, NERExact, ...) operate
  entirely in-memory on the full chunk list and touch Qdrant exactly once,
  in bulk, AFTER filtering decisions are made -- never during them. This
  module ports that pattern into CACD's Stage 1: chunk embeddings already
  decided KEEP are kept in a plain in-memory numpy array (a few thousand
  x 384 floats is a few MB -- trivial), and Stage 1 becomes a vectorized
  matrix-multiply top-K search (query_vecs @ pool_vecs.T) instead of a
  network/disk round trip. Qdrant itself is untouched during the decision
  loop; stage3_decision.py now upserts everything ONCE at the end of a
  config's ingest, purely to make the final kept set available for the
  downstream RAG retrieval evaluation step -- exactly like the other
  (non-CACD) baseline filters already do.

  Complexity trade-off (explicit, matches the paper's own note that
  SIMILARITY's O(N^2) cost is accepted for its dataset sizes): this makes
  Stage 1 an exact O(pool_size) search per chunk rather than the paper's
  originally-assumed O(log n) HNSW lookup, so cumulative Stage 1 cost is
  O(N^2) over a full ingest, same asymptotic class as the SIMILARITY
  baseline. In wall-clock terms this is still dramatically faster here
  because the per-call constant is now a single BLAS matmul in RAM instead
  of a disk-backed SQLite round trip -- at these corpus sizes (~5K-11K
  chunks, 384-dim) the full cumulative cost is well under a second. This is
  a genuine change from the paper's original Big-O framing (Section III-B)
  and should be described as such if reported, not silently substituted.

Public API:
  InMemoryIndex(dim)      — one instance per chunking-strategy config,
                             mirrors the lifecycle of one Qdrant collection.
  .top_k(query_vecs, k)   — batched top-K search, returns the SAME candidate
                             dict shape as batch_coarse_retrieve() so Stage 2
                             / Stage 3 / the guards need no changes at all.
  .add(chunks, vecs)      — append newly-KEPT chunks to the pool (call this
                             once per micro-batch, after Stage 3's decisions
                             are known -- same timing as the old per-batch
                             Qdrant upsert, just without the round trip).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """
    Defensive L2 normalization so a plain dot product equals cosine
    similarity. Mirrors rag_bench's Similarity filter, which does the same
    "should already be normalized but just in case" renormalization -- the
    embedder is expected to output unit vectors, but Stage 1's correctness
    (and CACD's redundancy decisions downstream) must not silently depend
    on that never changing.
    """
    if vecs.size == 0:
        return vecs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return vecs / norms


class InMemoryIndex:
    """
    Growing in-memory pool of (vector, metadata) for chunks already decided
    KEEP in the current config's ingest run. One instance per chunking
    config -- create a fresh one exactly where the old code called
    ensure_collection(cname, recreate=True), i.e. once per config, not once
    per corpus.
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
            list of length m; each entry is a list of up to k candidate
            dicts, sorted by descending cosine score, with the SAME fields
            batch_coarse_retrieve() produced from Qdrant hits:
            chunk_id, doc_id, title, text, score, parent_id, level.
            Empty list for a query if the pool is currently empty --
            matches the old "collection has no points yet" behaviour.
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
        after Stage 3 has decided which chunks in that batch are KEEP/MERGE
        -- matches the staleness contract already documented in
        stage3_decision.run_cacd_dedup: chunks within the same micro-batch
        never see each other, only chunks added by earlier batches do.
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
