"""
CACD Stage 3 — Decision.

Role: decide whether to drop or keep a new chunk based on two signals
produced by Stage 2: prob_duplicate (cross-encoder output) and NIS (Novel
Information Score from the attention matrix). Also orchestrates the full
Stage 1 -> Stage 2 -> Stage 3 pipeline for a batch of chunks.

Decision logic (3 zones), applied per candidate and combined by majority
vote across the K candidates (see _vote_decision):
  Zone 1 (prob >= PROB_HIGH): model is confident it is a duplicate.
    => DROP, subject to the length-aware guard.
  Zone 2 (prob <= PROB_LOW): model is confident it is NOT a duplicate.
    => KEEP immediately.
  Zone 3 (PROB_LOW < prob < PROB_HIGH): uncertainty zone.
    => NIS decides: NIS < NIS_DROP_THRESHOLD => DROP, otherwise KEEP.
    The length-aware guard also applies in this zone.

PROB_HIGH and PROB_LOW are derived from the Bayes-optimal cutoff
(calibration.py), not hand-picked constants.

Public API:
  run_cacd_dedup(...) -> (kept_chunks, audit_log)
"""

from __future__ import annotations

import logging
import math
import time as _time

from configs.settings import (
    CACD_COST_FALSE_NEGATIVE,
    CACD_COST_FALSE_POSITIVE,
    CACD_INGEST_BATCH_SIZE,
    CACD_TOP_K_CANDIDATES,
)
from src.dedup.calibration import bayes_optimal_cutoff
from src.dedup.stage1_inmemory_retrieval import InMemoryIndex
from src.dedup.stage2_cross_attention import (
    score_pairs_batched,
    _is_parent_child_pair,
    _strip_contextual_header,
)

logger = logging.getLogger(__name__)

# PROB_HIGH: prob_duplicate >= this value => DROP immediately (model is confident)
# PROB_LOW : prob_duplicate <= this value => KEEP immediately (model is confident)
# [PROB_LOW, PROB_HIGH]: uncertainty zone, NIS decides
_cutoff   = bayes_optimal_cutoff(CACD_COST_FALSE_POSITIVE, CACD_COST_FALSE_NEGATIVE)
PROB_HIGH = min(0.95, _cutoff + 0.3)
PROB_LOW  = max(0.05, _cutoff - 0.3)

# Midpoint of the normalized NIS entropy scale [0, 1]; used as the decision
# boundary inside the uncertainty zone.
NIS_DROP_THRESHOLD = 0.8

# Long chunks are assumed to carry more unique information and get a higher
# false-drop risk. A chunk longer than LENGTH_GUARD characters is protected
# from being dropped unless NIS < NIS_FLOOR.
LENGTH_GUARD = 300   # characters
NIS_FLOOR    = 0.3


def run_cacd_dedup(
    chunks: list[dict],
    dense_vecs: list[list[float]],
    cname: str,
    config_name: str,
    embed_fn=None,
    chunk_size: int = 400,
    micro_batch_size: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run the full CACD pipeline (Stage 1 -> Stage 2 -> Stage 3) for one
    chunking configuration, ingesting `chunks` into collection `cname`.

    Stage 1 searches an in-memory pool of already-KEPT chunk embeddings
    (src/dedup/stage1_inmemory_retrieval.py), not Qdrant. Qdrant is used
    exactly once per config, in a single bulk upsert at the end of this
    function, purely to make the final kept set available for the
    downstream retrieval evaluation step.

    Chunks are processed in micro-batches of `micro_batch_size` (default
    CACD_INGEST_BATCH_SIZE). Within a micro-batch:
      1. Stage 1 retrieves candidates for every chunk in the batch against
         the in-memory pool as it stood at the start of the batch (chunks
         within the same batch cannot see each other; a smaller
         micro_batch_size reduces this staleness window, with
         micro_batch_size=1 removing it entirely at the cost of more,
         smaller cross-encoder calls).
      2. All (chunk, candidate) pairs for the whole batch are scored in as
         few cross-encoder forward passes as possible via
         score_pairs_batched.
      3. Stage 3 runs the per-chunk majority vote (_vote_decision).
      4. Kept chunks are added to the in-memory pool for later batches to
         see, and accumulated for the single end-of-run Qdrant upsert.

    Args:
        chunks           : list of chunk dicts with embeddings pre-computed.
        dense_vecs       : embedding vector for each chunk (same order).
        cname            : Qdrant collection name; must already exist
                           (created by the caller via ensure_collection()).
        config_name      : benchmark config label, used only for logging.
        embed_fn         : unused by the current pipeline; kept for
                           interface compatibility with callers that pass it.
        chunk_size       : configured target chunk size (chars) for the
                           current chunking strategy; unused by the current
                           pipeline, kept for interface compatibility.
        micro_batch_size : chunks scored together per Stage 2 forward-pass
                           batch. Defaults to CACD_INGEST_BATCH_SIZE.

    Returns:
        (kept_chunks, audit_log)
    """
    from configs.settings import TEXT_EMBED_DIM
    from src.ingestion.vector_store import upsert_chunks

    if micro_batch_size is None:
        micro_batch_size = CACD_INGEST_BATCH_SIZE

    kept_chunks: list[dict] = []
    audit_log:   list[dict] = []

    # Fresh in-memory index per config, same lifecycle as the Qdrant
    # collection it replaces for Stage 1 retrieval during the decision loop.
    inmem_index = InMemoryIndex(dim=TEXT_EMBED_DIM)
    all_upsert_chunks: list[dict] = []
    all_upsert_vecs:   list[list[float]] = []

    # Cumulative wall-clock time per stage, logged at the end. Coarse
    # (measured around each stage's block, not per-chunk) so the overhead
    # of measuring it is negligible.
    _t_stage1  = 0.0
    _t_stage2  = 0.0
    _t_stage3  = 0.0
    _t_pool_add = 0.0
    _t_final_upsert = 0.0
    _t_other   = 0.0

    def _vote_decision(chunk: dict, valid_scored: list[dict]):
        """
        Majority vote across a chunk's valid (post-guard) candidates.
        Each candidate votes KEEP or DROP using the 3-zone rule plus the
        length-aware guard; tallying stops as soon as either side reaches
        a majority (ceil(len(valid_scored) / 2)). Defaults to KEEP if no
        majority is reached.

        Returns (decision, reason, deciding_candidate).
        """
        chunk_len  = len(chunk["text"])
        majority   = math.ceil(len(valid_scored) / 2)
        votes_drop = 0
        votes_keep = 0
        decision   = None
        reason     = ""
        best       = None

        for cand in valid_scored:
            prob = cand["prob_duplicate"]
            nis  = cand["nis_b_given_a"]

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

        if decision is None:
            decision = "keep"
            reason   = f"voting_no_majority (drop={votes_drop}, keep={votes_keep}) => default keep"
            best     = valid_scored[-1]

        return decision, reason, best

    n = len(chunks)

    for batch_start in range(0, n, micro_batch_size):
        batch_end    = min(batch_start + micro_batch_size, n)
        batch_chunks = chunks[batch_start:batch_end]
        batch_vecs   = dense_vecs[batch_start:batch_end]

        # Stage 1: retrieve candidates against the in-memory pool as it
        # stood at the start of this batch.
        _t0 = _time.perf_counter()
        candidates_list = inmem_index.top_k(batch_vecs, k=CACD_TOP_K_CANDIDATES)
        _t_stage1 += _time.perf_counter() - _t0
        if not candidates_list:
            candidates_list = [[] for _ in batch_chunks]

        # Build the flat (text_a, text_b) pair list for Stage 2, applying
        # guards (parent-child skip, header strip) before scoring.
        pending_status:  list[str]        = []  # "no_candidates" | "all_skipped" | "scored"
        pending_valid:   list[list[dict]] = []
        flat_pairs:      list[tuple[str, str]] = []
        flat_owner:      list[int] = []
        flat_cand:       list[dict] = []

        _t0 = _time.perf_counter()
        for bi, chunk in enumerate(batch_chunks):
            cands = candidates_list[bi] if bi < len(candidates_list) else []
            if not cands:
                pending_status.append("no_candidates")
                pending_valid.append([])
                continue

            valid_for_chunk = [
                cand for cand in cands
                if not _is_parent_child_pair(
                    chunk.get("parent_id"), chunk.get("level"),
                    cand.get("parent_id"), cand.get("level"),
                    cand["chunk_id"],
                )
            ]
            if not valid_for_chunk:
                pending_status.append("all_skipped")
                pending_valid.append([])
                continue

            pending_status.append("scored")
            pending_valid.append(valid_for_chunk)
            a_clean = _strip_contextual_header(chunk["text"])
            for cand in valid_for_chunk:
                b_clean = _strip_contextual_header(cand["text"])
                flat_pairs.append((a_clean, b_clean))
                flat_owner.append(bi)
                flat_cand.append(cand)

        _t_other += _time.perf_counter() - _t0

        # Stage 2: score the whole batch in as few forward passes as possible.
        _t0 = _time.perf_counter()
        flat_results = score_pairs_batched(flat_pairs) if flat_pairs else []
        _t_stage2 += _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        scored_per_chunk: list[list[dict]] = [[] for _ in batch_chunks]
        for owner_bi, cand, result in zip(flat_owner, flat_cand, flat_results):
            merged = dict(cand)
            merged.update(result)
            merged["redundancy_signal"] = min(
                result["coverage_a_to_b"], result["coverage_b_to_a"]
            )
            scored_per_chunk[owner_bi].append(merged)
        _t_other += _time.perf_counter() - _t0

        # Stage 3: per-chunk decision, then collect what needs to be
        # added to the index.
        to_upsert_chunks: list[dict] = []
        to_upsert_vecs:   list[list[float]] = []

        for bi, chunk in enumerate(batch_chunks):
            vec         = batch_vecs[bi]
            global_i    = batch_start + bi
            status      = pending_status[bi]

            if status == "no_candidates":
                kept_chunks.append(chunk)
                to_upsert_chunks.append(chunk)
                to_upsert_vecs.append(vec)
                audit_log.append({
                    "chunk_id":          chunk["chunk_id"],
                    "decision":          "keep",
                    "reason":            "no_candidates",
                    "best_p_duplicate":  0.0,
                    "best_candidate_id": "",
                })
                continue

            if status == "all_skipped":
                kept_chunks.append(chunk)
                to_upsert_chunks.append(chunk)
                to_upsert_vecs.append(vec)
                audit_log.append({
                    "chunk_id":           chunk["chunk_id"],
                    "decision":           "keep",
                    "reason":             "all_candidates_skipped",
                    "best_p_duplicate":   0.0,
                    "best_candidate_id":  "",
                    "nis_b_given_a":      1.0,
                    "coverage_a_to_b":    0.0,
                    "coverage_b_to_a":    0.0,
                    "redundancy_signal":  0.0,
                    "prob_high":          round(PROB_HIGH, 4),
                    "prob_low":           round(PROB_LOW, 4),
                    "nis_threshold":      NIS_DROP_THRESHOLD,
                })
                continue

            valid_scored = scored_per_chunk[bi]
            _t0 = _time.perf_counter()
            decision, reason, best = _vote_decision(chunk, valid_scored)
            _t_stage3 += _time.perf_counter() - _t0

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
                to_upsert_chunks.append(chunk)
                to_upsert_vecs.append(vec)

            if (global_i + 1) % 50 == 0:
                n_dropped = (global_i + 1) - len(kept_chunks)
                logger.info(
                    "  CACD progress: %d/%d | kept=%d | dropped=%d | "
                    "prob_range=[%.2f,%.2f] | nis_thresh=%.2f | batch_size=%d",
                    global_i + 1, n, len(kept_chunks), n_dropped,
                    PROB_LOW, PROB_HIGH, NIS_DROP_THRESHOLD, micro_batch_size,
                )

        # Add this batch's kept chunks to the in-memory pool, and
        # accumulate them for the single end-of-run Qdrant upsert.
        if to_upsert_chunks:
            _t0 = _time.perf_counter()
            inmem_index.add(to_upsert_chunks, to_upsert_vecs)
            _t_pool_add += _time.perf_counter() - _t0
            all_upsert_chunks.extend(to_upsert_chunks)
            all_upsert_vecs.extend(to_upsert_vecs)

    # Single bulk upsert for the whole config, so the kept set is
    # available for the downstream retrieval evaluation step. Batched at
    # up to 5000 points per call to keep the number of Qdrant round trips
    # small.
    if all_upsert_chunks:
        _t0 = _time.perf_counter()
        upsert_chunks(
            cname, all_upsert_chunks, all_upsert_vecs,
            batch_size=min(len(all_upsert_chunks), 5000),
        )
        _t_final_upsert = _time.perf_counter() - _t0

    logger.info(
        "  CACD done: %d => %d chunks kept "
        "(prob_range=[%.2f,%.2f], nis_thresh=%.2f, batch_size=%d)",
        len(chunks), len(kept_chunks),
        PROB_LOW, PROB_HIGH, NIS_DROP_THRESHOLD, micro_batch_size,
    )

    _t_total = _t_stage1 + _t_stage2 + _t_stage3 + _t_pool_add + _t_final_upsert + _t_other
    from configs.settings import DEVICE as _device
    try:
        import torch as _torch
        _cuda_ok = _torch.cuda.is_available()
        _gpu_name = _torch.cuda.get_device_name(0) if _cuda_ok else "N/A"
    except Exception:
        _cuda_ok, _gpu_name = None, "unknown"
    logger.info(
        "  CACD timing breakdown (%.1fs total): "
        "Stage1(in-memory retrieval)=%.1fs (%.0f%%) | "
        "Stage2(cross-encoder)=%.1fs (%.0f%%) | "
        "Stage3(voting)=%.1fs (%.0f%%) | "
        "pool_add=%.1fs (%.0f%%) | "
        "final_upsert(1x, Qdrant)=%.1fs (%.0f%%) | other=%.1fs (%.0f%%)",
        _t_total,
        _t_stage1, 100 * _t_stage1 / max(_t_total, 1e-9),
        _t_stage2, 100 * _t_stage2 / max(_t_total, 1e-9),
        _t_stage3, 100 * _t_stage3 / max(_t_total, 1e-9),
        _t_pool_add, 100 * _t_pool_add / max(_t_total, 1e-9),
        _t_final_upsert, 100 * _t_final_upsert / max(_t_total, 1e-9),
        _t_other,  100 * _t_other  / max(_t_total, 1e-9),
    )
    logger.info(
        "  Device: configs.settings.DEVICE=%s | "
        "torch.cuda.is_available()=%s | GPU=%s",
        _device, _cuda_ok, _gpu_name,
    )
    if _device == "cpu" or _cuda_ok is False:
        logger.warning(
            "  Running on CPU, not GPU; cross-encoder scoring will be "
            "substantially slower than on a GPU-equipped machine."
        )

    return kept_chunks, audit_log
