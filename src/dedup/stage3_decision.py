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
from src.dedup.calibration import RunningLogitCalibrator, bayes_optimal_cutoff
from src.dedup.stage1_coarse_retrieval import batch_coarse_retrieve
from src.dedup.stage2_cross_attention import score_candidates

logger = logging.getLogger(__name__)

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

    calibrator = RunningLogitCalibrator()
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
                "chunk_id": chunk["chunk_id"],
                "decision": "keep",
                "reason":   "no_candidates",
                "best_p_duplicate": 0.0,
                "best_candidate_id": "",
            })
            continue

        # Stage 2 — cross-attention scoring cho từng candidate
        scored = score_candidates(chunk["text"], candidates)

        # ── Tín hiệu redundancy đúng bản chất ───────────────────────────
        # raw_logit của ms-marco-MiniLM-L-6-v2 là RELEVANCE score (passage
        # B có liên quan tới query A không), KHÔNG phải DUPLICATE score.
        # Hai chunk cùng chủ đề (rất phổ biến trong 1 document SQuAD) có
        # thể có raw_logit rất cao dù nội dung hoàn toàn khác nhau — dùng
        # trực tiếp raw_logit làm tín hiệu dedup sẽ drop oan hàng loạt.
        #
        # Tín hiệu đúng hơn: redundancy_signal = min(coverage_a_to_b,
        # coverage_b_to_a) — một cặp chỉ thực sự "trùng lặp" khi CẢ HAI
        # chiều đều có độ bao phủ cao (A được B bao phủ nhiều VÀ B được A
        # bao phủ nhiều). Dùng min (không phải mean) để tránh trường hợp
        # 1 chiều bao phủ cao (B chứa trọn A, A là tập con của B) trong
        # khi chiều kia thấp bị tính nhầm thành "trùng lặp đối xứng".
        for s in scored:
            s["redundancy_signal"] = min(s["coverage_a_to_b"], s["coverage_b_to_a"])
            calibrator.update(s["redundancy_signal"])
        for s in scored:
            s["p_duplicate"] = round(
                calibrator.calibrated_probability(s["redundancy_signal"]), 4
            )

        best = max(scored, key=lambda s: s["p_duplicate"])

        # Stage 3 — quyết định threshold-free (chỉ nhánh Drop)
        if best["p_duplicate"] > CUTOFF:
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
            "reason":             "cross_attention_cutoff",
            "best_p_duplicate":   best["p_duplicate"],
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
            logger.info(
                "  CACD progress: %d/%d chunks xử lý | kept=%d | calib=%s",
                i + 1, len(chunks), len(kept_chunks), calibrator.stats(),
            )

    logger.info(
        "  CACD done: %d → %d chunks giữ lại (cutoff=%.4f)",
        len(chunks), len(kept_chunks), CUTOFF,
    )
    return kept_chunks, audit_log
