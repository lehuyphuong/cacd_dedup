"""
Qdrant vector store wrapper -- dense vectors only.

Role: store the final set of kept chunks per chunking configuration, so
they are available for the retrieval-quality evaluation step. Not used
during the CACD decision loop itself (see src/dedup/stage1_inmemory_retrieval.py).

Defaults to an embedded (in-process) Qdrant client, requiring no server.
Set configs.settings.QDRANT_URL to connect to a real Qdrant server instead.

Collection name format:
  "{COLLECTION_PREFIX}_{strategy}_{size}_{overlap}__cacd"
  e.g. "cacd_dedup_Contextual_300_0__cacd"

Payload fields stored per chunk:
  chunk_id, doc_id, title, text, char_start, char_end, parent_id, level
  (parent_id/level are used by Stage 2's parent-child guard)
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
    """Lazily create and return the Qdrant client (server or embedded,
    depending on configs.settings.QDRANT_URL)."""
    global _client
    if _client is None:
        if QDRANT_URL:
            logger.info("Connecting to Qdrant server at: %s", QDRANT_URL)
            _client = QdrantClient(url=QDRANT_URL)
        else:
            logger.info("Opening Qdrant embedded store at: %s", QDRANT_PATH)
            _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def collection_name(strategy: str, chunk_size: int, overlap: int, filter_tag: str) -> str:
    """Build a stable collection name for one (chunking strategy x filter) config."""
    base = f"{COLLECTION_PREFIX}_{strategy}_{chunk_size}_{overlap}"
    if filter_tag:
        return f"{base}__{filter_tag}"
    return base


def ensure_collection(cname: str, recreate: bool = True) -> str:
    """Create (or recreate) a Qdrant collection with cosine-distance vectors.

    Sets optimizers_config.indexing_threshold=0 so the collection is
    indexed immediately rather than only above Qdrant's default 20,000-point
    threshold, since collections here are typically smaller than that."""
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
) -> None:
    """
    Upsert chunks into a collection, `batch_size` points per call.

    Input:
        cname      : target collection name.
        chunks     : chunk dicts (must include chunk_id, doc_id, title,
                     text, char_start, char_end; parent_id/level optional).
        dense_vecs : embedding vector for each chunk (same order as chunks).
        batch_size : points per Qdrant upsert call.
    """
    client = get_client()
    for i in range(0, len(chunks), batch_size):
        batch_c = chunks[i : i + batch_size]
        batch_d = dense_vecs[i : i + batch_size]

        points = [
            PointStruct(
                id=abs(hash(chunk["chunk_id"])) % (2 ** 53),
                vector=dvec,
                payload={
                    "chunk_id":   chunk["chunk_id"],
                    "doc_id":     chunk["doc_id"],
                    "title":      chunk["title"],
                    "text":       chunk["text"],
                    "char_start": chunk["char_start"],
                    "char_end":   chunk["char_end"],
                    "parent_id":  chunk.get("parent_id"),
                    "level":      chunk.get("level"),
                },
            )
            for chunk, dvec in zip(batch_c, batch_d)
        ]
        client.upsert(collection_name=cname, points=points, wait=True)

    logger.info("Upserted %d points into '%s'", len(chunks), cname)


def collection_stats(cname: str) -> dict:
    """
    Return point count and disk size for a collection.

    Input:
        cname : collection name.

    Returns dict:
        collection   : cname
        points_count : number of points in the collection
        disk_size_du : human-readable size string (from `du -sh`)
        disk_mb      : numeric size in MB (sum of file sizes on disk)
    """
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
        "disk_size_du": disk_size_str,
        "disk_mb":      disk_mb,
    }


def delete_collection(cname: str) -> None:
    """Delete a collection to free disk space after benchmarking."""
    client   = get_client()
    existing = [c.name for c in client.get_collections().collections]
    if cname in existing:
        client.delete_collection(cname)
        logger.info("Deleted collection '%s'", cname)