"""
CACD Stage 3 — Quyết định threshold-free (CHỈ nhánh DROP, không Merge
trong phạm vi triển khai này — theo quyết định của user, Merge để
dành cho experiment sau).

P(duplicate) > cutoff  →  Bỏ qua, KHÔNG insert (drop)
Ngược lại               →  Insert vào Qdrant

cutoff được suy ra từ Bayes-optimal cost ratio (calibration.py),
không phải một hằng số đoán mò như cosine threshold 0.8 trước đây.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configs.settings import (
    CACD_COST_FALSE_NEGATIVE,
    CACD_COST_FALSE_POSITIVE,
    HEATMAP_DIR,
)
from src.dedup.stage1_coarse_retrieval import batch_coarse_retrieve
from src.dedup.stage2_cross_attention import score_candidates

logger = logging.getLogger(__name__)

# Ngưỡng tự nhiên của msmarco-MiniLM-L6-en-de-v1:
# prob_duplicate = sigmoid(raw_logit) — model được train để
# phân biệt duplicate/non-duplicate trực tiếp, nên 0.5 là ranh giới
# quyết định có căn cứ từ quá trình training, không phải số đoán mò.
# Bayes-optimal cost ratio vẫn được dùng để điều chỉnh nếu cần ưu
# tiên precision (tránh drop nhầm) hơn recall (tránh giữ thừa).
from src.dedup.calibration import bayes_optimal_cutoff
CUTOFF = bayes_optimal_cutoff(
    cost_false_positive=CACD_COST_FALSE_POSITIVE,
    cost_false_negative=CACD_COST_FALSE_NEGATIVE,
)


def save_heatmap(
    score_result: dict,
    chunk_id: str,
    candidate_id: str,
    config_name: str,
    decision: str,
) -> str:
    """
    Vẽ heatmap attention matrix giữa chunk mới và candidate, lưu vào
    results/heatmaps/{config_name}/ để phân tích trực quan đoạn nào
    trùng (Stage 2b "trích xuất attention matrix").
    """
    out_dir = HEATMAP_DIR / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    attn = score_result["attention_matrix"]
    sep_idx = score_result["sep_idx"]
    n_tokens = score_result["n_tokens"]

    # Vùng A (chunk mới) x Vùng B (candidate) — đúng phần mà
    # _max_alignment_coverage dùng để tính redundancy signal.
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)
    sub_attn = attn[a_range, b_range]

    tokens_a = score_result["tokens_a"]
    tokens_b = score_result["tokens_b"]

    if sub_attn.size == 0 or len(tokens_a) == 0 or len(tokens_b) == 0:
        return ""

    fig_w = max(6, min(0.3 * len(tokens_b), 20))
    fig_h = max(4, min(0.3 * len(tokens_a), 16))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(sub_attn, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(tokens_b)))
    ax.set_xticklabels(tokens_b, rotation=90, fontsize=6)
    ax.set_yticks(range(len(tokens_a)))
    ax.set_yticklabels(tokens_a, fontsize=6)
    ax.set_xlabel("Candidate (B)")
    ax.set_ylabel("Chunk mới (A)")
    ax.set_title(
        f"Cross-attention redundancy map\n"
        f"P(dup)={score_result.get('p_duplicate', 0):.3f} | "
        f"cov(A→B)={score_result['coverage_a_to_b']:.3f} | "
        f"cov(B→A)={score_result['coverage_b_to_a']:.3f} | "
        f"decision={decision}",
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention weight")
    fig.tight_layout()

    safe_chunk = chunk_id.replace("/", "_")[:40]
    safe_cand = candidate_id.replace("/", "_")[:40]
    out_path = out_dir / f"{safe_chunk}__vs__{safe_cand}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    return str(out_path)


def run_cacd_dedup(
    chunks: list[dict],
    dense_vecs: list[list[float]],
    cname: str,
    config_name: str,
    save_heatmaps: bool = True,
    max_heatmaps: int = 30,
) -> tuple[list[dict], list[dict]]:
    """
    Chạy đầy đủ pipeline CACD (Stage 1 → 2 → 3) cho 1 batch chunks
    chuẩn bị ingest vào collection `cname`.

    Chunks được xử lý TUẦN TỰ trong vòng lặp (mỗi chunk: coarse
    retrieve candidate từ index hiện có → cross-attention score →
    quyết định drop/keep → nếu keep thì insert NGAY để chunk tiếp
    theo trong cùng document cũng có thể bị phát hiện trùng với nó).

    Returns:
        (kept_chunks, audit_log)
        kept_chunks: list chunk được giữ lại (để caller insert vào Qdrant).
        audit_log  : list dict ghi lại quyết định cho từng chunk
                     (để phân tích / ghi CSV).
    """
    from src.ingestion.vector_store import upsert_chunks

    kept_chunks: list[dict] = []
    audit_log: list[dict] = []
    n_heatmaps_saved = 0

    for i, (chunk, vec) in enumerate(zip(chunks, dense_vecs)):
        # Stage 1 — coarse retrieval trên index ĐÃ TỒN TẠI (persistent,
        # bao gồm cả chunk vừa insert ở các vòng lặp trước trong cùng batch).
        candidates_list, _ = batch_coarse_retrieve(
            [chunk], [vec], cname, top_k=None,
        )
        candidates = candidates_list[0] if candidates_list else []

        if not candidates:
            # Không có gì để so sánh — chắc chắn không trùng lặp.
            kept_chunks.append(chunk)
            upsert_chunks(cname, [chunk], [vec])
            audit_log.append({
                "chunk_id":          chunk["chunk_id"],
                "decision":          "keep",
                "reason":            "no_candidates",
                "best_p_duplicate":  0.0,
                "best_candidate_id": "",
            })
            continue

        # Stage 2 — cross-attention scoring cho từng candidate.
        # score_candidates trả về prob_duplicate đã là sigmoid(raw_logit)
        # từ model msmarco-MiniLM-L6-en-de-v1 — model này được train để
        # phân biệt duplicate/non-duplicate trực tiếp, nên prob_duplicate
        # là tín hiệu đúng bản chất, không cần qua z-score calibrator.
        scored = score_candidates(chunk["text"], candidates)

        # Tính redundancy_signal bổ sung (coverage-based) để ghi audit log
        # và vẽ heatmap — KHÔNG dùng để quyết định drop/keep nữa.
        for s in scored:
            s["redundancy_signal"] = min(
                s["coverage_a_to_b"], s["coverage_b_to_a"]
            )

        # Chọn candidate có prob_duplicate cao nhất để quyết định.
        best = max(scored, key=lambda s: s["prob_duplicate"])

        # Stage 3 — quyết định dựa trên prob_duplicate với ngưỡng tự nhiên.
        # CUTOFF = 0.5 khi cost_FP = cost_FN (mặc định đối xứng), tương
        # đương ngưỡng sigmoid tự nhiên của model. Tăng cost_FP trong
        # settings.py để hệ thống thận trọng hơn khi drop.
        if best["prob_duplicate"] > CUTOFF:
            decision = "drop"
        else:
            decision = "keep"

        if save_heatmaps and n_heatmaps_saved < max_heatmaps:
            save_heatmap(
                best, chunk["chunk_id"], best["chunk_id"],
                config_name, decision,
            )
            n_heatmaps_saved += 1

        audit_log.append({
            "chunk_id":           chunk["chunk_id"],
            "decision":           decision,
            "reason":             "prob_duplicate_cutoff",
            "best_p_duplicate":   best["prob_duplicate"],
            "best_candidate_id":  best["chunk_id"],
            "coverage_a_to_b":    best["coverage_a_to_b"],
            "coverage_b_to_a":    best["coverage_b_to_a"],
            "redundancy_signal":  best["redundancy_signal"],
            "attn_entropy":       best["attn_entropy"],
            "cutoff_used":        round(CUTOFF, 4),
        })

        if decision == "keep":
            kept_chunks.append(chunk)
            upsert_chunks(cname, [chunk], [vec])

        if (i + 1) % 50 == 0:
            n_dropped = (i + 1) - len(kept_chunks)
            logger.info(
                "  CACD progress: %d/%d chunks xử lý | kept=%d | dropped=%d | cutoff=%.4f",
                i + 1, len(chunks), len(kept_chunks), n_dropped, CUTOFF,
            )

    logger.info(
        "  CACD done: %d → %d chunks giữ lại (cutoff=%.4f)",
        len(chunks), len(kept_chunks), CUTOFF,
    )
    return kept_chunks, audit_log
