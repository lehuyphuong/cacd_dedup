"""
CACD Stage 3 — Decision (DROP branch only).

Role: decide whether to drop or keep a new chunk based on two signals
produced by Stage 2: prob_duplicate (cross-encoder output) and
NIS (Novel Information Score from the attention matrix).

Decision logic — 3 zones:
  Zone 1 (prob >= PROB_HIGH): model is confident it is a duplicate.
    => DROP, subject to the length-aware guard.
  Zone 2 (prob <= PROB_LOW): model is confident it is NOT a duplicate.
    => KEEP immediately.
  Zone 3 (PROB_LOW < prob < PROB_HIGH): uncertainty zone.
    => NIS decides: NIS < NIS_DROP_THRESHOLD => DROP, otherwise KEEP.
    The length-aware guard also applies in this zone.

PROB_HIGH and PROB_LOW are derived from the Bayes-optimal cutoff
(calibration.py), not hand-picked constants.
NIS_DROP_THRESHOLD = 0.8 is the saturation point observed empirically
on SQuAD: values above 0.8 produce identical results due to LENGTH_GUARD
controlling most decisions.
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

# ── Decision thresholds ───────────────────────────────────────────────────────
#
# PROB_HIGH: prob_duplicate >= this value => DROP immediately (model is confident)
# PROB_LOW : prob_duplicate <= this value => KEEP immediately (model is confident)
# [PROB_LOW, PROB_HIGH]: uncertainty zone => NIS decides
#
# Defaults: PROB_HIGH=0.8, PROB_LOW=0.2 create an uncertainty zone of [0.2, 0.8].
# Adjust via CACD_COST_FP/FN in settings.py:
#   CUTOFF = bayes_optimal_cutoff(cost_FP, cost_FN) => used as the midpoint
#   PROB_HIGH = min(0.95, CUTOFF + 0.3)
#   PROB_LOW  = max(0.05, CUTOFF - 0.3)
_cutoff   = bayes_optimal_cutoff(CACD_COST_FALSE_POSITIVE, CACD_COST_FALSE_NEGATIVE)
PROB_HIGH = min(0.95, _cutoff + 0.3)
PROB_LOW  = max(0.05, _cutoff - 0.3)

# NIS_DROP_THRESHOLD: natural midpoint of the normalised entropy scale [0, 1].
# Based on SemDeDup (Abbas et al., 2023): retaining 80-85% of data yields
# the best retrieval quality. Empirically saturates at 0.8 on SQuAD — values
# above 0.8 produce identical results because LENGTH_GUARD controls most decisions.
NIS_DROP_THRESHOLD = 0.8

# Length-aware guard — inspired by ROOTS (Laurençon et al., 2023):
# long chunks carry more unique information and have a higher false-positive rate.
# Do not drop a chunk longer than LENGTH_GUARD characters unless NIS < NIS_FLOOR.
LENGTH_GUARD = 300   # characters
NIS_FLOOR    = 0.3   # absolute floor: NIS < NIS_FLOOR => drop even if chunk is long


def save_heatmap(
    score_result: dict,
    chunk_id: str,
    candidate_id: str,
    config_name: str,
    decision: str,
) -> str:
    """
    Save an attention-matrix heatmap between the new chunk and a candidate.

    Output: results/heatmaps/{config_name}/{chunk_id}__vs__{candidate_id}.png

    Input:
        score_result : dict returned by score_pair (must contain attention_matrix,
                       sep_idx, n_tokens, tokens_a, tokens_b).
        chunk_id     : ID of the new chunk (A).
        candidate_id : ID of the candidate (B).
        config_name  : benchmark config label (used as sub-directory).
        decision     : "drop" or "keep" (shown in the plot title).

    Returns path to the saved PNG, or "" if the matrix is empty.
    """
    out_dir = HEATMAP_DIR / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    attn     = score_result["attention_matrix"]
    sep_idx  = score_result["sep_idx"]
    n_tokens = score_result["n_tokens"]

    # Cross-attention sub-matrix: rows = A tokens, columns = B tokens
    a_range  = slice(1, sep_idx)
    b_range  = slice(sep_idx + 1, n_tokens - 1)
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
    ax.set_ylabel("New chunk (A)")
    ax.set_title(
        f"Cross-attention redundancy map\n"
        f"P(dup)={score_result.get('prob_duplicate', 0):.3f} | "
        f"NIS={score_result.get('nis_b_given_a', 0):.3f} | "
        f"cov(A=>B)={score_result['coverage_a_to_b']:.3f} | "
        f"cov(B=>A)={score_result['coverage_b_to_a']:.3f} | "
        f"decision={decision}",
        fontsize=8,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention weight")
    fig.tight_layout()

    safe_chunk = chunk_id.replace("/", "_")[:40]
    safe_cand  = candidate_id.replace("/", "_")[:40]
    out_path   = out_dir / f"{safe_chunk}__vs__{safe_cand}.png"
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
    Run the full CACD pipeline (Stage 1 => 2 => 3) for one batch of chunks
    being ingested into collection `cname`.

    Chunks are processed sequentially: for each chunk, Stage 1 retrieves
    candidates from the current index, Stage 2 scores them, Stage 3 decides
    drop or keep, and kept chunks are inserted immediately so subsequent
    chunks in the same document can be detected as duplicates of them.

    Input:
        chunks      : list of chunk dicts with embeddings pre-computed.
        dense_vecs  : embedding vector for each chunk (same order).
        cname       : Qdrant collection name (empty at the start of ingest).
        config_name : benchmark config label (used for heatmap paths and logs).
        save_heatmaps: whether to save attention heatmaps (disabled by default
                       to reduce ingest time).
        max_heatmaps : maximum number of heatmaps to save per config.

    Returns:
        (kept_chunks, audit_log)
        kept_chunks : chunks that were kept (inserted into Qdrant).
        audit_log   : list of dicts recording the decision for each chunk.
    """
    from src.ingestion.vector_store import upsert_chunks

    kept_chunks: list[dict] = []
    audit_log:   list[dict] = []
    n_heatmaps_saved = 0

    for i, (chunk, vec) in enumerate(zip(chunks, dense_vecs)):
        # Stage 1 — coarse retrieval on the CURRENT index (includes chunks
        # inserted in earlier iterations of the same ingest pass).
        candidates_list, _ = batch_coarse_retrieve(
            [chunk], [vec], cname, top_k=None,
        )
        candidates = candidates_list[0] if candidates_list else []

        if not candidates:
            # No neighbours in the index yet => definitely not a duplicate.
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

        # Stage 2 — cross-attention scoring.
        # Pass parent_id and level to skip parent-child pairs, and strip
        # the contextual header before scoring to avoid false redundancy.
        scored = score_candidates(
            chunk["text"],
            candidates,
            chunk_parent_id=chunk.get("parent_id"),
            chunk_level=chunk.get("level"),
        )

        # Redundancy signal for audit log (not used in the decision).
        for s in scored:
            s["redundancy_signal"] = min(
                s["coverage_a_to_b"], s["coverage_b_to_a"]
            )

        # Exclude skipped parent-child candidates before selecting the best.
        valid_scored = [s for s in scored if not s.get("skipped", False)]

        if not valid_scored:
            # All candidates were parent-child pairs => nothing to dedup against.
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

        # Stage 3 — Early-exit majority voting across K candidates.
        #
        # Each valid candidate votes independently DROP or KEEP using the
        # same 3-zone decision rule + length-aware guard as before.
        # Voting stops as soon as one side reaches ceil(K/2) votes.
        #
        # Rationale: the original design dropped a chunk whenever the single
        # highest-scoring candidate triggered DROP, even if K-1 other
        # candidates all voted KEEP. A single Stage 1 retrieval artifact
        # (a false positive from HNSW) could therefore cause a false drop.
        # Majority voting requires genuine consensus before committing to DROP.

        import math
        chunk_len  = len(chunk["text"])
        majority   = math.ceil(len(valid_scored) / 2)
        votes_drop = 0
        votes_keep = 0
        decision   = None
        reason     = ""
        best       = None   # candidate that cast the deciding vote

        for cand in valid_scored:
            prob = cand["prob_duplicate"]
            nis  = cand["nis_b_given_a"]

            # Same 3-zone + length-aware guard logic as before, applied per candidate
            if prob >= PROB_HIGH:
                if chunk_len > LENGTH_GUARD and nis > NIS_FLOOR:
                    vote        = "keep"
                    vote_reason = f"length_guard ({chunk_len}chars > {LENGTH_GUARD}, nis={nis:.3f})"
                else:
                    vote        = "drop"
                    vote_reason = f"prob_high ({prob:.3f} >= {PROB_HIGH})"
            elif prob <= PROB_LOW:
                vote        = "keep"
                vote_reason = f"prob_low ({prob:.3f} <= {PROB_LOW})"
            else:
                if nis < NIS_DROP_THRESHOLD:
                    if chunk_len > LENGTH_GUARD:
                        vote        = "keep"
                        vote_reason = f"length_guard_uncertainty ({chunk_len}chars, nis={nis:.3f})"
                    else:
                        vote        = "drop"
                        vote_reason = f"nis_low ({nis:.3f} < {NIS_DROP_THRESHOLD}, prob={prob:.3f})"
                else:
                    vote        = "keep"
                    vote_reason = f"nis_high ({nis:.3f} >= {NIS_DROP_THRESHOLD}, prob={prob:.3f})"

            if vote == "drop":
                votes_drop += 1
            else:
                votes_keep += 1

            # Early exit: first side to reach majority wins
            if votes_drop >= majority:
                decision = "drop"
                reason   = f"voting_drop ({votes_drop}/{len(valid_scored)} >= {majority}) | deciding: {vote_reason}"
                best     = cand
                break
            if votes_keep >= majority:
                decision = "keep"
                reason   = f"voting_keep ({votes_keep}/{len(valid_scored)} >= {majority}) | deciding: {vote_reason}"
                best     = cand
                break

        # Fallback: no majority reached (e.g. only 1 candidate after guard filtering)
        # => default KEEP to avoid silent data loss
        if decision is None:
            decision = "keep"
            reason   = f"voting_no_majority (drop={votes_drop}, keep={votes_keep}) => default keep"
            best     = valid_scored[-1]

        # Heatmap saving is disabled to reduce ingest time.
        # Uncomment for visual analysis:
        # if save_heatmaps and n_heatmaps_saved < max_heatmaps:
        #     save_heatmap(
        #         best, chunk["chunk_id"], best["chunk_id"],
        #         config_name, decision,
        #     )
        #     n_heatmaps_saved += 1

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
        "  CACD done: %d => %d chunks kept "
        "(prob_range=[%.2f,%.2f], nis_thresh=%.2f)",
        len(chunks), len(kept_chunks),
        PROB_LOW, PROB_HIGH, NIS_DROP_THRESHOLD,
    )
    return kept_chunks, audit_log
