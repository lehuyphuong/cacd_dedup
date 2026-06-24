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
from src.dedup.calibration import bayes_optimal_cutoff
from src.dedup.stage1_coarse_retrieval import batch_coarse_retrieve
from src.dedup.stage2_cross_attention import score_candidates

logger = logging.getLogger(__name__)

# ── Ngưỡng quyết định Stage 3 ────────────────────────────────────────────────
#
# PROB_HIGH: prob_duplicate >= giá trị này → DROP ngay (model rất chắc)
# PROB_LOW : prob_duplicate <= giá trị này → KEEP ngay (model rất chắc)
# Vùng [PROB_LOW, PROB_HIGH]: uncertainty zone → NIS quyết định
#
# Mặc định: PROB_HIGH=0.8, PROB_LOW=0.2 tạo ra vùng uncertainty [0.2, 0.8]
# Điều chỉnh bằng cách thay đổi CACD_COST_FP/FN trong settings.py:
#   CUTOFF = bayes_optimal_cutoff(cost_FP, cost_FN) → dùng làm PROB_HIGH
#   1 - CUTOFF → dùng làm PROB_LOW (đối xứng)
_cutoff  = bayes_optimal_cutoff(CACD_COST_FALSE_POSITIVE, CACD_COST_FALSE_NEGATIVE)
PROB_HIGH = min(0.95, _cutoff + 0.3)   # vd. 0.5 + 0.3 = 0.8
PROB_LOW  = max(0.05, _cutoff - 0.3)   # vd. 0.5 - 0.3 = 0.2

# NIS_DROP_THRESHOLD: ranh giới tự nhiên của thang entropy chuẩn hóa [0,1]
# 0.5 = entropy trung bình của token B bằng 50% maximum entropy lý thuyết
# → B "ít thông tin mới hơn nửa" so với trường hợp hoàn toàn khác A
NIS_DROP_THRESHOLD = 0.5


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
        f"P(dup)={score_result.get('prob_duplicate', 0):.3f} | "
        f"NIS={score_result.get('nis_b_given_a', 0):.3f} | "
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
        # Truyền parent_id và level để skip parent-child pairs,
        # và strip contextual header trước khi score.
        scored = score_candidates(
            chunk["text"],
            candidates,
            chunk_parent_id=chunk.get("parent_id"),
            chunk_level=chunk.get("level"),
        )

        # Tính redundancy_signal bổ sung để ghi audit log.
        for s in scored:
            s["redundancy_signal"] = min(
                s["coverage_a_to_b"], s["coverage_b_to_a"]
            )

        # Lọc bỏ candidates đã bị skip (parent-child) trước khi quyết định
        valid_scored = [s for s in scored if not s.get("skipped", False)]

        if not valid_scored:
            # Toàn bộ candidates đều là parent/child → không có gì để dedup
            kept_chunks.append(chunk)
            upsert_chunks(cname, [chunk], [vec])
            audit_log.append({
                "chunk_id":          chunk["chunk_id"],
                "decision":          "keep",
                "reason":            "all_candidates_skipped",
                "best_p_duplicate":  0.0,
                "best_candidate_id": "",
                "nis_b_given_a":     1.0,
                "coverage_a_to_b":   0.0,
                "coverage_b_to_a":   0.0,
                "redundancy_signal": 0.0,
                "prob_high":         round(PROB_HIGH, 4),
                "prob_low":          round(PROB_LOW, 4),
                "nis_threshold":     NIS_DROP_THRESHOLD,
            })
            continue

        # Chọn candidate có prob_duplicate cao nhất trong số hợp lệ.
        best = max(valid_scored, key=lambda s: s["prob_duplicate"])

        # ── Stage 3 — Quyết định dựa trên NIS (Novel Information Score) ─────
        #
        # Thay vì dùng 1 threshold cố định trên prob_duplicate, kết hợp
        # prob_duplicate với NIS để xử lý đúng các trường hợp partial
        # overlap (50%, 60%...) mà binary threshold không giải quyết được:
        #
        # Logic 3 vùng:
        #
        #  Vùng 1 — prob_duplicate cao (model rất chắc là duplicate):
        #    → DROP ngay, không cần hỏi NIS
        #    → NIS có thể không đáng tin ở đây vì model đã rất tự tin
        #
        #  Vùng 2 — prob_duplicate thấp (model rất chắc là NOT duplicate):
        #    → KEEP ngay, không cần hỏi NIS
        #
        #  Vùng 3 — prob_duplicate ở giữa (partial overlap, model không chắc):
        #    → NIS quyết định:
        #      NIS thấp → B ít thông tin mới so với A → DROP
        #      NIS cao  → B có nhiều thông tin mới    → KEEP
        #
        # Ngưỡng NIS_DROP_THRESHOLD: không phải số đặt tay tùy tiện.
        # Dựa trên nền tảng Information Theory: NIS = 0.5 nghĩa là
        # "entropy trung bình của token B bằng 50% maximum entropy lý thuyết"
        # — tức là token B chỉ "phân tán" sang 50% không gian token A
        # → B không hoàn toàn mới, nhưng cũng không hoàn toàn giống A.
        # Chọn 0.5 làm ranh giới tự nhiên của thang entropy chuẩn hóa.
        #
        # Các ngưỡng xác suất (PROB_HIGH, PROB_LOW) có thể điều chỉnh
        # trong settings.py mà không cần sửa code.

        prob = best["prob_duplicate"]
        nis  = best["nis_b_given_a"]

        if prob >= PROB_HIGH:
            decision = "drop"
            reason   = f"prob_high ({prob:.3f} >= {PROB_HIGH})"
        elif prob <= PROB_LOW:
            decision = "keep"
            reason   = f"prob_low ({prob:.3f} <= {PROB_LOW})"
        else:
            # Vùng không chắc chắn → NIS quyết định
            if nis < NIS_DROP_THRESHOLD:
                decision = "drop"
                reason   = f"nis_low ({nis:.3f} < {NIS_DROP_THRESHOLD}, prob={prob:.3f})"
            else:
                decision = "keep"
                reason   = f"nis_high ({nis:.3f} >= {NIS_DROP_THRESHOLD}, prob={prob:.3f})"

        if save_heatmaps and n_heatmaps_saved < max_heatmaps:
            save_heatmap(
                best, chunk["chunk_id"], best["chunk_id"],
                config_name, decision,
            )
            n_heatmaps_saved += 1

        audit_log.append({
            "chunk_id":           chunk["chunk_id"],
            "decision":           decision,
            "reason":             reason,
            "best_p_duplicate":   best["prob_duplicate"],
            "best_candidate_id":  best["chunk_id"],
            "nis_b_given_a":      best["nis_b_given_a"],
            "coverage_a_to_b":    best["coverage_a_to_b"],
            "coverage_b_to_a":    best["coverage_b_to_a"],
            "redundancy_signal":  best["redundancy_signal"],
            "prob_high":          round(PROB_HIGH, 4),
            "prob_low":           round(PROB_LOW, 4),
            "nis_threshold":      NIS_DROP_THRESHOLD,
        })

        if decision == "keep":
            kept_chunks.append(chunk)
            upsert_chunks(cname, [chunk], [vec])

        if (i + 1) % 50 == 0:
            n_dropped = (i + 1) - len(kept_chunks)
            logger.info(
                "  CACD progress: %d/%d | kept=%d | dropped=%d | "
                "prob_range=[%.2f,%.2f] | nis_thresh=%.2f",
                i + 1, len(chunks), len(kept_chunks), n_dropped,
                PROB_LOW, PROB_HIGH, NIS_DROP_THRESHOLD,
            )

    logger.info(
        "  CACD done: %d → %d chunks giữ lại "
        "(prob_range=[%.2f,%.2f], nis_thresh=%.2f)",
        len(chunks), len(kept_chunks),
        PROB_LOW, PROB_HIGH, NIS_DROP_THRESHOLD,
    )
    return kept_chunks, audit_log
