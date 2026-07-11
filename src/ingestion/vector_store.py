"""
Qdrant in-process (embedded) vector store — dense only.

Uses QdrantClient(path=...) — no Docker, no port, no root.

API compatibility note:
  Older Qdrant used named vectors via VectorsConfig(dense=VectorParams(...)).
  Newer Qdrant (>=1.9) deprecated that Union syntax. We now use the simpler
  unnamed vector API: vectors_config=VectorParams(...) directly, and pass
  plain list[float] as the vector in PointStruct and query_points.

Collection name format:
  "{COLLECTION_PREFIX}_{strategy}_{size}_{overlap}__cacd"
  e.g. "cacd_dedup_Contextual_300_0__cacd"

Payload fields stored per chunk:
  chunk_id, doc_id, title, text, char_start, char_end,
  parent_id, level  (used by Stage 2 to skip parent-child pairs)
"""

from __future__ import annotations

import logging
import subprocess

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)

from configs.settings import (
    COLLECTION_PREFIX,
    QDRANT_PATH,
    QDRANT_URL,
    TEXT_EMBED_DIM,
)

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            logger.info("Connecting to Qdrant server at: %s", QDRANT_URL)
            _client = QdrantClient(url=QDRANT_URL)
        else:
            logger.info(
                "Opening Qdrant embedded store at: %s "
                "(local/brute-force mode -- set configs.settings.QDRANT_URL "
                "to use a real server instead; see comment there)",
                QDRANT_PATH,
            )
            _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def collection_name(strategy: str, chunk_size: int, overlap: int, filter_tag: str) -> str:
    """Stable collection name for one (chunker x filter) config."""
    base = f"{COLLECTION_PREFIX}_{strategy}_{chunk_size}_{overlap}"
    if filter_tag:
        return f"{base}__{filter_tag}"
    return base


def ensure_collection(cname: str, recreate: bool = True) -> str:
    """Create (or recreate) a Qdrant collection."""
    client   = get_client()
    existing = [c.name for c in client.get_collections().collections]

    if cname in existing:
        if recreate:
            logger.info("Dropping collection '%s'", cname)
            client.delete_collection(cname)
        else:
            logger.info("Collection '%s' already exists — skipping.", cname)
            return cname

    client.create_collection(
        collection_name=cname,
        vectors_config=VectorParams(
            size=TEXT_EMBED_DIM,
            distance=Distance.COSINE,
        ),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        # Qdrant only builds a real HNSW graph for a segment once it holds
        # more than optimizers_config.indexing_threshold points (default
        # 20,000) -- below that it deliberately does a brute-force scan
        # instead, on the assumption a graph isn't worth building yet.
        # Our per-config collections here are ~5K-11K points, so under the
        # default they NEVER get indexed and every Stage 1 query is an
        # O(n) scan over everything ingested so far -- exactly the
        # growing per-batch slowdown observed (6ms/chunk -> 60ms/chunk
        # over one 9022-chunk run). Lowering the threshold forces Qdrant
        # to index early so retrieval is the intended O(log n) HNSW
        # lookup instead. 0 = index immediately, no minimum segment size.
        optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
    )
    logger.info(
        "Created collection '%s' (dense=%d-dim cosine)", cname, TEXT_EMBED_DIM
    )
    return cname


def upsert_chunks(
    cname: str,
    chunks: list[dict],
    dense_vecs: list[list[float]],
    batch_size: int = 256,
    wait: bool = True,
) -> None:
    """
    Upsert chunks into the collection in batches.

    Args:
        wait: passed straight through to client.upsert(). True (default,
              unchanged behaviour) blocks until Qdrant confirms the write
              is durable/visible before returning. False skips that
              confirmation -- if this turns out to actually be faster in
              Local Mode, it means the "wait" step itself has overhead
              beyond the raw write; if it makes no difference, Local
              Mode's synchronous, single-process design means there was
              nothing to skip. Either way, when wait=False this function
              verifies the point count itself afterward (with a short
              retry loop) before returning, so callers relying on an
              immediate read right after (e.g. the RAG eval step) are not
              exposed to a read-before-write race even if Qdrant's own
              wait mechanism was skipped.
    """
    client = get_client()
    for i in range(0, len(chunks), batch_size):
        batch_c = chunks[i : i + batch_size]
        batch_d = dense_vecs[i : i + batch_size]

        points = [
            PointStruct(
                id=abs(hash(chunk["chunk_id"])) % (2 ** 53),
                vector=dvec,   # unnamed vector — plain list[float]
                payload={
                    "chunk_id":   chunk["chunk_id"],
                    "doc_id":     chunk["doc_id"],
                    "title":      chunk["title"],
                    "text":       chunk["text"],
                    "char_start": chunk["char_start"],
                    "char_end":   chunk["char_end"],
                    # Stored for HierarchicalParentChild parent-child skip guard
                    "parent_id":  chunk.get("parent_id"),
                    "level":      chunk.get("level"),
                },
            )
            for chunk, dvec in zip(batch_c, batch_d)
        ]
        client.upsert(collection_name=cname, points=points, wait=wait)

    if not wait:
        _verify_point_count(client, cname, expected_at_least=len(chunks))

    logger.info("Upserted %d points into '%s'", len(chunks), cname)


def _verify_point_count(
    client: QdrantClient, cname: str, expected_at_least: int,
    max_wait_s: float = 5.0, poll_interval_s: float = 0.1,
) -> None:
    """
    Safety net for wait=False: poll the collection's reported point count
    until it reaches at least `expected_at_least`, or give up after
    max_wait_s and log a warning (does not raise -- callers decide what to
    do with stale data, this only makes the risk visible instead of silent).
    """
    import time as _time
    deadline = _time.perf_counter() + max_wait_s
    while _time.perf_counter() < deadline:
        count = client.get_collection(cname).points_count
        if count is not None and count >= expected_at_least:
            return
        _time.sleep(poll_interval_s)
    logger.warning(
        "upsert_chunks(wait=False): point count for '%s' did not reach "
        "%d within %.1fs -- downstream reads may see incomplete data. "
        "Consider reverting to wait=True.",
        cname, expected_at_least, max_wait_s,
    )


def collection_stats(cname: str) -> dict:
    """Return point count and disk size for a collection."""
    client = get_client()
    info   = client.get_collection(cname)

    col_path      = QDRANT_PATH / "collection" / cname
    disk_size_str = "0"
    disk_mb       = 0.0
    if col_path.exists():
        try:
            result = subprocess.run(
                ["du", "-sh", str(col_path)],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                disk_size_str = result.stdout.split("\t")[0].strip()
        except Exception:
            pass
        disk_bytes = sum(
            f.stat().st_size
            for f in col_path.rglob("*")
            if f.is_file()
        )
        disk_mb = round(disk_bytes / 1024 / 1024, 2)

    return {
        "collection":   cname,
        "points_count": info.points_count,
        "disk_size_du": disk_size_str,   # human-readable (du -sh output)
        "disk_mb":      disk_mb,          # numeric MB for CSV
    }


def delete_collection(cname: str) -> None:
    """Delete a collection to free disk space after benchmarking."""
    client   = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if cname in existing:
        client.delete_collection(cname)
        logger.info("Deleted collection '%s'", cname)