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
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configs.settings import (
    CACD_COST_FALSE_NEGATIVE,
    CACD_COST_FALSE_POSITIVE,
    CACD_ENABLE_MERGE,
    CACD_INGEST_BATCH_SIZE,
    CACD_TOP_K_CANDIDATES,
    HEATMAP_DIR,
    NIS_SENTENCE_NOVEL,
    MIN_NOVEL_CHARS,
    MERGE_SAME_DOC_ONLY,
    MERGE_MAX_SIZE_MULTIPLIER,
    MERGE_MAX_EMBED_CHARS,
)
from src.dedup.calibration import bayes_optimal_cutoff
from src.dedup.stage1_inmemory_retrieval import InMemoryIndex
from src.dedup.stage2_cross_attention import (
    score_candidates,
    score_pairs_batched,
    _is_parent_child_pair,
    _strip_contextual_header,
)

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


def _sentence_level_merge(
    chunk_text: str,
    chunk_doc_id: str,
    chunk_size: int,
    valid_scored: list[dict],
    embed_fn,
) -> tuple[str | None, dict | None, list[float] | None]:
    """
    Sentence-level merge: when chunk A is voted DROP, extract novel sentences
    from A and merge them into the best matching candidate already in the index.

    Algorithm:
      1. Split A into sentences.
      2. Restrict candidates to the SAME document as A (Fix 1 — see below).
      3. For each sentence sᵢ, score against the remaining candidates using
         CrossEncoder. sᵢ is novel if min(NIS(sᵢ | Bⱼ)) > NIS_SENTENCE_NOVEL.
         Uses MIN rule: sᵢ must be novel relative to every candidate.
      4. For each novel sentence, identify the merge target:
         target(sᵢ) = argmin_j NIS(sᵢ | Bⱼ) — the candidate with the lowest
         novelty score, i.e. the closest topical match among the remaining
         (same-document) candidates.
      5. Group novel sentences by merge target.
      6. For the target with the most novel sentences, append novel sentences
         one at a time up to a hard size cap (Fix 2 + Fix 3 — see below).
      7. If the resulting novel part is still >= MIN_NOVEL_CHARS: return
         B_merged. Otherwise: return None (nothing worth indexing).

    Fix 1 (MERGE_SAME_DOC_ONLY) — cross-document contamination:
      The persistent index intentionally spans the whole corpus (dedup must
      catch redundancy across documents, that scope stays unchanged). But
      MERGE specifically must not splice a sentence from document A into a
      chunk that keeps being served under document B's doc_id — otherwise
      Precision/IoU are evaluated against Te (tokens of ONE reference
      document) while Tr silently gains tokens from a different document.
      Root-cause analysis on the full-dataset merge run (Recall highest,
      Precision lowest, IoU lowest of all methods) traced back to this:
      target selection never checked doc_id equality.

    Fix 2 + 3 (MERGE_MAX_SIZE_MULTIPLIER / MERGE_MAX_EMBED_CHARS) — unbounded
    growth + silent embedding truncation:
      Previously there was only a MIN_NOVEL_CHARS floor and no ceiling, so a
      single "hub" chunk_id could absorb an unbounded number of merges over
      one ingest pass. Separately, embed_fn() truncates internally at the
      embedding model's max_seq_length — re-embedding after merge (kept
      below) only produces a vector that reflects the *whole* merged text if
      that text is short enough to avoid truncation in the first place.
      MERGE_MAX_EMBED_CHARS is a conservative character-based proxy for that
      token limit; MERGE_MAX_SIZE_MULTIPLIER keeps merged chunks proportional
      to the strategy's own target chunk_size. The effective cap is the
      smaller of the two.

    Args:
        chunk_text   : text of new chunk A being considered for DROP.
        chunk_doc_id : doc_id of A — used to restrict merge targets to the
                       same document (Fix 1).
        chunk_size   : configured target chunk size (chars) for the current
                       chunking strategy — used to scale the size cap (Fix 2).
        valid_scored : list of scored candidate dicts from Stage 2
                       (already filtered for parent-child pairs).
        embed_fn     : callable(list[str]) => list[list[float]]
                       used to re-embed B_merged after merge.

    Returns:
        (B_merged_chunk, B_merged_vec, target_B)  if merge is worthwhile
        (None, None, None)                         if no novel content found
    """
    import re
    from src.dedup.stage2_cross_attention import score_sentences_batched

    # Early exit (speed only, does not change any outcome): the accepted
    # novel_text built below is always a subset of chunk_text's own
    # sentences, so its length can never exceed len(chunk_text). If
    # chunk_text itself is already shorter than MIN_NOVEL_CHARS, no possible
    # combination of its sentences could ever pass the Step 7 floor check —
    # skip sentence splitting and all scoring entirely in that case.
    if len(chunk_text) < MIN_NOVEL_CHARS:
        return None, None, None

    # Step 1 — split A into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', chunk_text.strip()) if s.strip()]
    if not sentences:
        return None, None, None

    # Step 2 — Fix 1: restrict merge candidates to the same document as A.
    # (Stage 1/2/3's DROP decision still uses the full cross-document index —
    # only the MERGE target is constrained here.)
    if MERGE_SAME_DOC_ONLY:
        same_doc_candidates = [c for c in valid_scored if c.get("doc_id") == chunk_doc_id]
    else:
        same_doc_candidates = valid_scored

    if not same_doc_candidates:
        logger.info(
            "  Merge skipped: no same-document candidate among %d valid candidate(s) "
            "(doc_id=%s) => falling back to plain drop",
            len(valid_scored), chunk_doc_id,
        )
        return None, None, None

    # Step 3 & 4 — per sentence, compute NIS against the same-document candidates
    # novel(sᵢ) = True if min_j NIS(sᵢ | Bⱼ) > NIS_SENTENCE_NOVEL
    # target(sᵢ) = argmin_j NIS(sᵢ | Bⱼ)
    #
    # Note on direction: CrossEncoder(Bⱼ, sᵢ) sets Bⱼ as A and sᵢ as B,
    # so NIS measures "how much of sᵢ is novel relative to Bⱼ" — correct direction.
    #
    # Speed note: this loops over candidates (few — at most K, and typically
    # fewer once restricted to the same document) and, for each, scores ALL
    # sentences in a single batched forward pass via score_sentences_batched.
    # This computes the exact same (sentence, candidate) NIS values as calling
    # score_pair(cand["text"], sent) once per pair — only the batching
    # changes, not the math — turning m*k single-pair forward passes into
    # k batched ones.

    # nis_per_sentence[i] = list of (candidate, nis) for sentence i, one entry per candidate
    nis_per_sentence: list[list[tuple[dict, float]]] = [[] for _ in sentences]

    for cand in same_doc_candidates:
        batch_results = score_sentences_batched(cand["text"], sentences)
        for i, result in enumerate(batch_results):
            nis_per_sentence[i].append((cand, result["nis_b_given_a"]))

    novel_sentences_by_target: dict[str, list[str]] = {}  # {candidate_chunk_id: [sentences]}

    for sent, nis_scores in zip(sentences, nis_per_sentence):
        min_nis_cand, min_nis_val = min(nis_scores, key=lambda x: x[1])

        if min_nis_val > NIS_SENTENCE_NOVEL:
            # Sentence is novel relative to ALL same-doc candidates (min > threshold)
            target_id = min_nis_cand["chunk_id"]
            novel_sentences_by_target.setdefault(target_id, [])
            novel_sentences_by_target[target_id].append(sent)

    if not novel_sentences_by_target:
        return None, None, None

    # Step 5 — pick target with most novel sentences
    best_target_id = max(novel_sentences_by_target, key=lambda k: len(novel_sentences_by_target[k]))
    novel_sentences = novel_sentences_by_target[best_target_id]
    target_cand = next(c for c in same_doc_candidates if c["chunk_id"] == best_target_id)

    # Step 6 — Fix 2 + 3: cap how much novel text can be appended.
    # Effective cap = min(chunk_size * MERGE_MAX_SIZE_MULTIPLIER, MERGE_MAX_EMBED_CHARS),
    # counted against the CURRENT length of the target's own text so repeated
    # merges into the same target_cand over the ingest pass cannot exceed it.
    size_cap = min(chunk_size * MERGE_MAX_SIZE_MULTIPLIER, MERGE_MAX_EMBED_CHARS)
    budget   = max(0, int(size_cap) - len(target_cand["text"]) - 1)  # -1 for the joining space

    accepted: list[str] = []
    used = 0
    for sent in novel_sentences:
        add_len = len(sent) + (1 if accepted else 0)  # +1 for joining space
        if used + add_len > budget:
            break
        accepted.append(sent)
        used += add_len

    if not accepted:
        logger.info(
            "  Merge skipped: target '%s' already at/near size cap (%d chars, cap=%d) "
            "=> no room for novel content",
            best_target_id, len(target_cand["text"]), int(size_cap),
        )
        return None, None, None

    novel_text = " ".join(accepted)

    if len(novel_sentences) > len(accepted):
        logger.info(
            "  Merge: size cap reached — kept %d/%d novel sentence(s) for target '%s' "
            "(remaining sentence(s) fall back to plain drop, not merged elsewhere)",
            len(accepted), len(novel_sentences), best_target_id,
        )

    # Step 7 — min length guard
    if len(novel_text) < MIN_NOVEL_CHARS:
        logger.info(
            "  Merge skipped: novel_text too short (%d chars < %d)",
            len(novel_text), MIN_NOVEL_CHARS,
        )
        return None, None, None

    # Construct B_merged — keeps chunk_id of B so upsert overwrites B in Qdrant.
    # target_cand comes from Stage 1 payload and may not have char_start/char_end
    # (those fields are optional in the payload). Provide safe defaults so that
    # upsert_chunks does not raise KeyError.
    B_merged = {
        "chunk_id":   target_cand["chunk_id"],
        "doc_id":     target_cand["doc_id"],
        "title":      target_cand.get("title", ""),
        "text":       target_cand["text"] + " " + novel_text,
        "char_start": target_cand.get("char_start", 0),
        "char_end":   target_cand.get("char_end", 0),
        "parent_id":  target_cand.get("parent_id"),
        "level":      target_cand.get("level"),
    }

    # Re-embed B_merged. Because of the size cap above, B_merged["text"] is
    # guaranteed to stay within MERGE_MAX_EMBED_CHARS, so this re-embedding
    # is not silently truncated the way an unbounded merge would be.
    B_merged_vec = embed_fn([B_merged["text"]])[0]

    logger.info(
        "  Merge: appended %d novel sentence(s) (%d chars, same doc_id=%s) to candidate '%s' "
        "(new total %d chars, cap=%d)",
        len(accepted), len(novel_text), chunk_doc_id, best_target_id,
        len(B_merged["text"]), int(size_cap),
    )
    return B_merged, target_cand, B_merged_vec



def run_cacd_dedup(
    chunks: list[dict],
    dense_vecs: list[list[float]],
    cname: str,
    config_name: str,
    embed_fn=None,
    save_heatmaps: bool = True,
    max_heatmaps: int = 30,
    chunk_size: int = 400,
    micro_batch_size: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run the full CACD pipeline (Stage 1 => 2 => 3 + Merge) for one batch
    of chunks being ingested into collection `cname`.

    ARCHITECTURE CHANGE (see conversation history): Stage 1 no longer
    queries Qdrant. It searches an in-memory numpy pool of already-KEPT
    chunk embeddings instead (src/dedup/stage1_inmemory_retrieval.py).
    Qdrant is touched exactly ONCE per config, in bulk, at the very end of
    this function -- purely to make the final kept set available for the
    downstream RAG retrieval evaluation, the same pattern the non-CACD
    baseline filters (Similarity, NERExact, ...) already used. This was a
    deliberate trade-off after measuring that Qdrant's embedded/local
    client mode (SQLite-backed, brute-force by design, not HNSW) made
    per-chunk Stage 1 cost grow with collection size regardless of GPU
    speed, K, or Qdrant-side config -- see stage1_inmemory_retrieval.py's
    module docstring for the full writeup. This changes the paper's
    originally-assumed Big-O for Stage 1 (Section III-B) from O(log n)
    HNSW to O(pool_size) exact search, same asymptotic class as SIMILARITY;
    report it as such if these numbers go in the paper, not silently.

    Chunks are processed in MICRO-BATCHES of `micro_batch_size` (default
    CACD_INGEST_BATCH_SIZE), not one at a time. Within a micro-batch:
      1. Stage 1 retrieves candidates for every chunk in the batch against
         the in-memory pool as it stood at the START of the batch (no
         chunk in a batch sees another chunk from the SAME batch — see
         trade-off note below).
      2. All (chunk, candidate) pairs across the whole batch are flattened
         into one list and scored in as few cross-encoder forward passes
         as possible via score_pairs_batched, instead of one forward pass
         per chunk (previously: N chunks = N forward passes of K=5 pairs
         each; now: N chunks = ceil(N*K / sub_batch_size) forward passes).
      3. Stage 3's per-candidate vote + majority decision runs per chunk,
         identical logic to the previous fully-sequential version (see
         _vote_decision below — factored out but byte-for-byte the same
         rule as before).
      4. All KEPT (and, if enabled, merged) chunks in the batch are added
         to the in-memory pool so later micro-batches can see them, and
         accumulated into a running list that gets upserted into Qdrant
         ONCE, after the whole loop finishes (see end of this function).

    Trade-off (staleness within a batch): because Stage 1 retrieval for the
    whole batch happens before any of the batch's decisions are known,
    two near-duplicate chunks that both land in the SAME micro-batch will
    not detect each other (each only sees chunks already indexed before
    the batch started). Smaller micro_batch_size shrinks this blind spot;
    micro_batch_size=1 reproduces the exact old fully-sequential behaviour.
    This is a deliberate speed/staleness trade-off, not a bug — see
    Section III-G limitations discussion for the analogous trade-off in
    the (currently disabled) merge step.

    When DROP is decided and CACD_ENABLE_MERGE is True, sentence-level
    merge is attempted exactly as before (unchanged, still processed one
    chunk at a time since merge is off by default and not the bottleneck
    this restructuring targets — see Section III-G / CACD_ENABLE_MERGE).

    Args:
        chunks           : list of chunk dicts with embeddings pre-computed.
        dense_vecs       : embedding vector for each chunk (same order).
        cname            : Qdrant collection name (empty at start of ingest;
                           collection itself is still created by the caller
                           via ensure_collection() as before).
        config_name      : benchmark config label (for heatmap paths/logs).
        embed_fn         : callable(list[str]) => list[list[float]].
                           Required for sentence-level merge to re-embed
                           B_merged. If None, merge is skipped.
        save_heatmaps    : whether to save attention heatmaps.
        max_heatmaps     : maximum heatmaps per config.
        chunk_size       : configured target chunk size (chars) for the
                           current chunking strategy — used to scale the
                           merge size cap.
        micro_batch_size : chunks scored together per Stage 2 forward-pass
                           batch. Defaults to CACD_INGEST_BATCH_SIZE.

    Returns:
        (kept_chunks, audit_log)
    """
    from configs.settings import TEXT_EMBED_DIM
    from src.ingestion.vector_store import upsert_chunks
    import time as _time

    if micro_batch_size is None:
        micro_batch_size = CACD_INGEST_BATCH_SIZE

    kept_chunks: list[dict] = []
    audit_log:   list[dict] = []

    # In-memory Stage 1 index for this config (fresh pool per config, same
    # lifecycle as the Qdrant collection it replaces during the decision
    # loop). Also accumulate everything to upsert into Qdrant ONCE at the
    # end, instead of once per micro-batch.
    inmem_index = InMemoryIndex(dim=TEXT_EMBED_DIM)
    all_upsert_chunks: list[dict] = []
    all_upsert_vecs:   list[list[float]] = []

    # ── Diagnostic timing (added to root-cause an unexplained ingest-time
    # regression -- see conversation). Cumulative wall-clock time spent in
    # each stage across the whole ingest run, printed at the end. This is
    # deliberately coarse (perf_counter around each stage's code block, not
    # per-chunk) so it adds negligible overhead of its own and can be left
    # on without skewing the very numbers it's trying to measure.
    _t_stage1  = 0.0   # Stage 1: in-memory numpy top-K search (was: Qdrant query)
    _t_stage2  = 0.0   # Stage 2: score_pairs_batched (cross-encoder forward passes)
    _t_stage3  = 0.0   # Stage 3: per-chunk voting (should be ~free, CPU only)
    _t_pool_add = 0.0  # appending newly-kept chunks into the in-memory pool
    _t_final_upsert = 0.0  # ONE bulk Qdrant upsert at the very end
    _t_other   = 0.0   # guard filtering / bookkeeping between stages

    def _vote_decision(chunk: dict, valid_scored: list[dict]):
        """
        Stage 3 — early-exit majority voting across K candidates. Identical
        rule to the original fully-sequential implementation (Eq. 6 / the
        Vote(A,B) function): each valid candidate votes KEEP/DROP using the
        3-zone decision + length-aware guard, tallied with early exit at
        ceil(|valid_scored| / 2). Factored out here so the same exact logic
        runs whether chunks are processed one at a time or in a batch —
        this refactor changes nothing about the decision itself.
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

        # ── Stage 1 (whole batch, in-memory pool as it stood at batch start) ──
        _t0 = _time.perf_counter()
        candidates_list = inmem_index.top_k(batch_vecs, k=CACD_TOP_K_CANDIDATES)
        _t_stage1 += _time.perf_counter() - _t0
        if not candidates_list:
            candidates_list = [[] for _ in batch_chunks]

        # ── Build the flat (text_a, text_b) pair list for Stage 2,
        # applying the same guards as before (parent-child skip, header
        # strip) BEFORE scoring so guarded-out pairs never reach the model ──
        pending_status:  list[str]       = []   # "no_candidates" | "all_skipped" | "scored"
        pending_valid:   list[list[dict]] = []  # valid (post-guard) candidates per chunk
        flat_pairs:      list[tuple[str, str]] = []
        flat_owner:      list[int] = []         # index into batch_chunks
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

        # ── Stage 2 (whole batch, as few forward passes as possible) ────
        # NOTE: on the very FIRST micro-batch of the whole run, this also
        # includes lazy model loading (get_cross_encoder()) + the
        # last-layer-eager self-test forward pass -- a one-time cost, not
        # representative of steady-state per-batch Stage 2 time. Compare
        # _t_stage2 across the printed per-N-chunks progress lines if you
        # want to isolate that one-time cost from the recurring cost.
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

        # ── Stage 3 (per chunk, same rule as always) + collect upserts ──
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
            elif CACD_ENABLE_MERGE:
                if embed_fn is not None:
                    logger.info(
                        "  Attempting merge for dropped chunk '%s' (best_p=%.3f)",
                        chunk["chunk_id"], best["prob_duplicate"],
                    )
                    B_merged, target_cand, B_merged_vec = _sentence_level_merge(
                        chunk["text"], chunk["doc_id"], chunk_size, valid_scored, embed_fn,
                    )
                    if B_merged is not None:
                        to_upsert_chunks.append(B_merged)
                        to_upsert_vecs.append(B_merged_vec)
                        audit_log[-1]["decision"]    = "merge"
                        audit_log[-1]["reason"]      += f" | merged into '{target_cand['chunk_id']}'"
                        audit_log[-1]["merged_into"] = target_cand["chunk_id"]
                        logger.info(
                            "  MERGE done: chunk '%s' merged into '%s'",
                            chunk["chunk_id"], target_cand["chunk_id"],
                        )
                    else:
                        logger.info(
                            "  MERGE skipped: no novel content found in chunk '%s' => pure drop",
                            chunk["chunk_id"],
                        )
            # else: CACD_ENABLE_MERGE is False => pure drop, nothing upserted.

            if (global_i + 1) % 50 == 0:
                n_dropped = (global_i + 1) - len(kept_chunks)
                logger.info(
                    "  CACD progress: %d/%d | kept=%d | dropped=%d | "
                    "prob_range=[%.2f,%.2f] | nis_thresh=%.2f | batch_size=%d",
                    global_i + 1, n, len(kept_chunks), n_dropped,
                    PROB_LOW, PROB_HIGH, NIS_DROP_THRESHOLD, micro_batch_size,
                )

        # ── Add this batch's kept/merged chunks to the in-memory pool (so
        # later micro-batches can retrieve against them, same timing as the
        # old per-batch Qdrant upsert) and accumulate for ONE bulk Qdrant
        # upsert after the whole loop finishes ──
        if to_upsert_chunks:
            _t0 = _time.perf_counter()
            inmem_index.add(to_upsert_chunks, to_upsert_vecs)
            _t_pool_add += _time.perf_counter() - _t0
            all_upsert_chunks.extend(to_upsert_chunks)
            all_upsert_vecs.extend(to_upsert_vecs)

    # ── ONE bulk upsert into Qdrant for the whole config, purely so the
    # final kept set is available for the downstream RAG retrieval
    # evaluation step -- Qdrant played no role in any decision above ──
    if all_upsert_chunks:
        _t0 = _time.perf_counter()
        upsert_chunks(cname, all_upsert_chunks, all_upsert_vecs)
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
        "  Device check: configs.settings.DEVICE=%s | "
        "torch.cuda.is_available()=%s | GPU=%s",
        _device, _cuda_ok, _gpu_name,
    )
    if _device == "cpu" or _cuda_ok is False:
        logger.warning(
            "  *** Running on CPU, not GPU -- this alone can be 10-50x "
            "slower than the GPU baseline and is the most likely explanation "
            "for an ingest time far above the ~88s/config baseline. ***"
        )

    return kept_chunks, audit_log