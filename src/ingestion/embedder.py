"""
Embedding layer — all-MiniLM-L6-v2 via sentence-transformers (384-dim).

Paper Section 3.4:
  "all-MiniLM-L6-v2 … chosen as the reference model because it offers
   a good balance between speed, reproducibility, and computational cost."

Dense-only (no BM25 sparse), matching paper's retrieval protocol.

Public API:
    get_model()                     → SentenceTransformer singleton
    embed_texts(texts)              → list[list[float]]  (384-dim each)
    embed_chunks_batched(chunks)    → generator of (chunk, dense_vec)
"""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np
from sentence_transformers import SentenceTransformer

from configs.settings import EMBED_BATCH_SIZE, EMBED_MODEL, TEXT_EMBED_DIM

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
        logger.info(
            "Model loaded — output dim: %d", _model.get_sentence_embedding_dimension()
        )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Returns list of 384-dim float vectors (L2-normalized by default in MiniLM).
    """
    if not texts:
        return []
    model = get_model()
    # show_progress_bar=False to keep logs clean during benchmarking
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,   # unit-norm → cosine sim = dot product
        convert_to_numpy=True,
    )
    return vecs.tolist()


def embed_chunks_batched(
    chunks: list[dict],
    batch_size: int = EMBED_BATCH_SIZE,
) -> Generator[tuple[dict, list[float]], None, None]:
    """
    Yields (chunk, dense_vector) for each chunk.
    Processes in batches to avoid OOM on large corpora.
    """
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        dense_vecs = embed_texts(texts)
        for chunk, dv in zip(batch, dense_vecs):
            yield chunk, dv
