"""
CACD Stage 1 — Coarse Retrieval (bi-encoder + HNSW).

Pipeline doc Section 4.1, "Đóng góp 1":
  Batch query Qdrant với toàn bộ m vector của document cùng lúc,
  lấy top-K ứng viên gần nhất cho mỗi chunk (K nhỏ, cố định).

Độ phức tạp: O(m log n) — không đổi về Big-O so với HNSW thuần túy.
Vai trò: thu hẹp candidate set từ n (toàn bộ index) xuống K (hằng số
nhỏ), để Stage 2 (cross-attention, đắt) chỉ cần xử lý K ứng viên
thay vì toàn bộ n.

Không dùng GPU song song hóa thật trong bản triển khai này (CPU-only
môi trường), nhưng kiến trúc batch-query vẫn được giữ nguyên — đây
là điểm có thể nâng cấp lên cuVS/FAISS-GPU sau này mà không đổi logic.
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
    Với mỗi chunk mới (đã có embedding sẵn), tìm top-K ứng viên gần
    nhất ĐÃ TỒN TẠI trong collection Qdrant (persistent index, không
    phải chỉ so sánh nội bộ batch).

    Args:
        chunks    : list chunk dict (chưa insert vào Qdrant).
        dense_vecs: embedding tương ứng từng chunk (đã tính sẵn).
        cname     : tên collection Qdrant đang ingest dần (đã có thể
                    chứa chunk từ các batch trước).
        top_k     : K ứng viên gần nhất lấy ra cho mỗi chunk. None =
                    dùng CACD_TOP_K_CANDIDATES từ settings.

    Returns:
        (candidates_per_chunk, elapsed_seconds)
        candidates_per_chunk[i] = list các candidate dict (đã có sẵn
        trong index) ứng với chunks[i], rỗng nếu collection chưa có
        điểm nào hoặc không tìm thấy candidate nào.
    """
    if top_k is None:
        top_k = CACD_TOP_K_CANDIDATES

    client = get_client()
    t0 = time.perf_counter()

    candidates_per_chunk: list[list[dict]] = []

    # Batch query — gửi toàn bộ m vector cùng lúc trong 1 request.
    # Qdrant xử lý từng vector độc lập trên cùng 1 graph HNSW cố định
    # (đã build từ các lần insert trước đó).
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
        # Collection rỗng hoặc chưa tồn tại — không có gì để so sánh.
        logger.debug("Coarse retrieve: collection trống hoặc lỗi (%s)", exc)
        results = [None] * len(dense_vecs)

    for batch_result in results:
        if batch_result is None:
            candidates_per_chunk.append([])
            continue
        points = getattr(batch_result, "points", batch_result)
        cands = []
        for hit in points:
            payload = hit.payload or {}
            cands.append({
                "chunk_id": payload.get("chunk_id", ""),
                "doc_id":   payload.get("doc_id", ""),
                "title":    payload.get("title", ""),
                "text":     payload.get("text", ""),
                "score":    round(hit.score, 4),
            })
        candidates_per_chunk.append(cands)

    elapsed = time.perf_counter() - t0
    n_with_candidates = sum(1 for c in candidates_per_chunk if c)
    logger.debug(
        "  Stage1 coarse retrieve: %d/%d chunks có candidate (top_k=%d) trong %.2fs",
        n_with_candidates, len(chunks), top_k, elapsed,
    )
    return candidates_per_chunk, elapsed
