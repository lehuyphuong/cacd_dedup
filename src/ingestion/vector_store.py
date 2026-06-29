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
    PointStruct,
    VectorParams,
)

from configs.settings import (
    COLLECTION_PREFIX,
    QDRANT_PATH,
    TEXT_EMBED_DIM,
)

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        logger.info("Opening Qdrant embedded store at: %s", QDRANT_PATH)
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
) -> None:
    """Upsert chunks into the collection in batches."""
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
        client.upsert(collection_name=cname, points=points, wait=True)

    logger.info("Upserted %d points into '%s'", len(chunks), cname)


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
