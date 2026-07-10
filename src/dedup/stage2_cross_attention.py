"""
CACD Stage 2 — Cross-Attention Redundancy Scorer (CARS).

Role: for each pair (new_chunk, candidate_i), produce a redundancy signal
by jointly encoding both texts through a cross-encoder and extracting
information from the resulting attention matrix.

Pipeline:
  2a. Joint encoding : [CLS] new_chunk [SEP] candidate_i [SEP]
  2b. Attention extraction : last layer, averaged across all heads
  2c. Coverage signals : max-alignment coverage (BERTScore-style)
  2d. Novel Information Score (NIS) : entropy of attention B => A

Model: cross-encoder/msmarco-MiniLM-L6-en-de-v1 (pretrained, no fine-tuning).
Complexity: O(K) forward passes per new chunk; K is a small constant
            (CACD_TOP_K_CANDIDATES) because Stage 1 already narrowed candidates.

Public API:
  score_pair(text_a, text_b)          => dict of scores for one pair
  score_candidates(chunk_text, cands) => list of scored candidate dicts
"""

from __future__ import annotations

import contextlib
import logging

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.settings import CACD_CROSS_ENCODER_MODEL, CACD_USE_FP16, DEVICE

logger = logging.getLogger(__name__)

_tokenizer = None
_model     = None


def _autocast_ctx():
    """
    Mixed-precision context for the cross-encoder forward pass.

    Only enabled on CUDA: torch.autocast on CPU does not speed anything up
    and can even be slower, so this is a deliberate no-op (nullcontext) for
    CPU-only runs regardless of CACD_USE_FP16.
    """
    if CACD_USE_FP16 and DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def get_cross_encoder():
    """Lazy-load cross-encoder model and tokenizer (pretrained, no fine-tuning)."""
    global _tokenizer, _model
    if _model is None:
        logger.info("Loading cross-encoder: %s", CACD_CROSS_ENCODER_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(CACD_CROSS_ENCODER_MODEL)
        _model     = AutoModelForSequenceClassification.from_pretrained(
            CACD_CROSS_ENCODER_MODEL,
            output_attentions=True,   # attention matrix required, not just logits
        )
        _model.to(DEVICE)
        _model.eval()
        logger.info("Cross-encoder loaded on %s", DEVICE)
    return _tokenizer, _model


def _max_alignment_coverage(
    attn: torch.Tensor,
    sep_idx: int,
    n_tokens: int,
) -> tuple[float, float]:
    """
    Compute max-alignment coverage from the attention matrix (BERTScore-style).

    For the last attention layer averaged across heads:
      coverage(A => B) = mean over tokens in A of the maximum attention weight
                         that token assigns to any token in B.
      coverage(B => A) = same, in the opposite direction.

    Args:
        attn    : attention matrix (n_tokens, n_tokens), heads already averaged.
        sep_idx : position of the first [SEP] token (A/B boundary).
        n_tokens: total number of real tokens (excluding padding).

    Returns:
        (coverage_a_to_b, coverage_b_to_a)
    """
    # Region A: tokens 1..sep_idx-1 (excluding [CLS] at position 0)
    # Region B: tokens sep_idx+1..n_tokens-2 (excluding final [SEP])
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)

    sub_a_to_b = attn[a_range, b_range]   # (len_A, len_B)
    sub_b_to_a = attn[b_range, a_range]   # (len_B, len_A)

    if sub_a_to_b.numel() == 0 or sub_b_to_a.numel() == 0:
        return 0.0, 0.0

    # Each token in A finds its best-matching token in B (max over B dimension), then average
    coverage_a_to_b = sub_a_to_b.max(dim=1).values.mean().item()
    coverage_b_to_a = sub_b_to_a.max(dim=1).values.mean().item()

    return coverage_a_to_b, coverage_b_to_a


def _novel_information_score(
    attn: torch.Tensor,
    sep_idx: int,
    n_tokens: int,
) -> float:
    """
    Novel Information Score (NIS) — measures how much new information B
    carries relative to A, based on the entropy of the attention B => A.

    Theoretical basis (Information Theory):
      attention B=>A[j, :] is a probability distribution over tokens in A,
      representing how much token j in B relies on tokens in A for context.

      Low entropy of attention B=>A[j]:
        token j focuses on 1-2 specific tokens in A
        => A can "explain" token j
        => token j carries little new information.

      High entropy of attention B=>A[j]:
        token j spreads attention uniformly across A
        => no token in A can explain token j
        => token j likely carries new information.

    Formula:
      1. Re-normalise sub_B=>A row-wise (removes softmax dilution from full sequence).
      2. H(j) = -sum_i p(i|j) * log(p(i|j))   per token j in B.
      3. NIS = mean(H(j)) / log(|A|)            normalised to [0, 1].

    Normalisation uses log(|A|) — the theoretical maximum entropy when token B
    spreads uniformly across all tokens in A. This is a natural scale from
    Information Theory, not a hand-picked constant.

    Args:
        attn    : attention matrix (n_tokens, n_tokens), heads already averaged.
        sep_idx : position of the first [SEP] token (A/B boundary).
        n_tokens: total number of real tokens.

    Returns:
        nis: float in [0, 1].
             NIS => 0: B is fully explained by A => DROP candidate.
             NIS => 1: B is entirely novel relative to A => KEEP.
    """
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)

    sub_b_to_a = attn[b_range, a_range]   # (len_B, len_A)

    if sub_b_to_a.numel() == 0:
        return 1.0   # nothing to compare => treat B as entirely novel

    len_a = sub_b_to_a.shape[1]
    if len_a < 2:
        return 1.0

    # Re-normalise attention B=>A over the A region only
    # (removes softmax dilution caused by attending to the full sequence)
    row_sums    = sub_b_to_a.sum(dim=1, keepdim=True).clamp(min=1e-9)
    prob_b_to_a = sub_b_to_a / row_sums   # (len_B, len_A), each row sums to 1

    # Entropy per token in B:  H(j) = -sum_i p(i|j) * log(p(i|j))
    ent_per_token = -(prob_b_to_a * torch.log(prob_b_to_a + 1e-9)).sum(dim=1)  # (len_B,)

    # Theoretical maximum entropy = log(len_A)
    # (achieved when token B spreads uniformly across all tokens in A)
    max_ent = float(np.log(len_a))

    if max_ent < 1e-9:
        return 1.0

    nis = float((ent_per_token / max_ent).mean().clamp(0.0, 1.0).item())
    return round(nis, 4)


@torch.no_grad()
def score_pair(text_a: str, text_b: str) -> dict:
    """
    Score the redundancy of one pair (text_a, text_b) via the cross-encoder.

    Input:
        text_a : new chunk (A).
        text_b : candidate chunk already in the index (B).

    Returns dict:
        raw_logit       : float  -- raw cross-encoder output
        prob_duplicate  : float  -- sigmoid(logit) for binary models;
                                    softmax[-1] for multi-class models
        coverage_a_to_b : float  -- fraction of A's content covered by B
        coverage_b_to_a : float  -- fraction of B's content covered by A
        nis_b_given_a   : float  -- Novel Information Score of B given A
    """
    tokenizer, model = get_cross_encoder()

    inputs = tokenizer(
        text_a, text_b,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(DEVICE)

    with _autocast_ctx():
        outputs = model(**inputs)

    # Handle both binary models (scalar logit) and multi-class models (e.g. NLI 3-class)
    logits = outputs.logits.squeeze()
    if logits.dim() == 0:
        raw_logit = logits.item()
        prob_dup  = float(torch.sigmoid(logits).item())
    else:
        # Last class is typically entailment / match / duplicate
        raw_logit = logits[-1].item()
        prob_dup  = float(torch.softmax(logits, dim=-1)[-1].item())

    # Last attention layer averaged across all heads.
    # The last layer carries the strongest semantic signal for the classification head.
    attentions      = outputs.attentions          # tuple of (1, num_heads, seq, seq)
    last_layer_attn = attentions[-1][0]           # (num_heads, seq, seq)
    avg_attn        = last_layer_attn.mean(dim=0) # (seq, seq)

    input_ids = inputs["input_ids"][0]
    tokens    = tokenizer.convert_ids_to_tokens(input_ids)
    n_tokens  = int(inputs["attention_mask"][0].sum().item())

    sep_token_id  = tokenizer.sep_token_id
    sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
    sep_idx       = int(sep_positions[0].item()) if len(sep_positions) > 0 else n_tokens // 2

    cov_a_to_b, cov_b_to_a = _max_alignment_coverage(avg_attn, sep_idx, n_tokens)
    nis = _novel_information_score(avg_attn, sep_idx, n_tokens)

    return {
        "raw_logit":        round(raw_logit, 4),
        "prob_duplicate":   round(prob_dup, 4),
        "coverage_a_to_b":  round(cov_a_to_b, 4),
        "coverage_b_to_a":  round(cov_b_to_a, 4),
        "nis_b_given_a":    nis,
        # attention_matrix and token lists disabled — heatmap off, avoids
        # serialising large numpy arrays during benchmarking.
        # Uncomment for visual analysis:
        # "attention_matrix": avg_attn[:n_tokens, :n_tokens].cpu().numpy(),
        # "tokens_a":         tokens[1:sep_idx],
        # "tokens_b":         tokens[sep_idx + 1:n_tokens - 1],
        # "sep_idx":          sep_idx,
        # "n_tokens":         n_tokens,
    }


@torch.no_grad()
def score_candidates_batched(
    chunk_text: str,
    candidates: list[dict],
    chunk_parent_id: str | None = None,
    chunk_level: str | None = None,
) -> list[dict]:
    """
    Batch version of score_candidates — tokenises all K candidates together
    and runs a single forward pass instead of K sequential passes.

    On GPU a single batched forward pass utilises parallelism far better than
    K sequential calls; expected speedup ~3-4x for K=5.

    Handles:
      - Stripping the contextual header before scoring.
      - Skipping parent-child pairs (no forward pass for those).
      - Computing NIS from the per-pair attention sub-matrix
        (attention is not shared across pairs, so per-pair extraction
        still runs inside the loop, but the GPU kernel is batched).

    Input:
        chunk_text     : text of the new chunk (A).
        candidates     : list of candidate dicts from Stage 1.
        chunk_parent_id: parent_id of the new chunk (for HPC guard).
        chunk_level    : level field of the new chunk (for HPC guard).

    Returns:
        list of candidate dicts, each extended with scoring fields
        (prob_duplicate, nis_b_given_a, coverage_*, skipped, skip_reason).
    """
    tokenizer, model = get_cross_encoder()
    text_a_clean = _strip_contextual_header(chunk_text)

    to_score: list[tuple[int, dict]] = []   # (original_idx, candidate)
    results:  list[dict]             = []

    for i, cand in enumerate(candidates):
        if _is_parent_child_pair(
            chunk_parent_id, chunk_level,
            cand.get("parent_id"), cand.get("level"),
            cand.get("chunk_id", ""),
        ):
            results.append({**cand,
                "skipped":        True,
                "skip_reason":    "parent_child_pair",
                "prob_duplicate": 0.0,
                "nis_b_given_a":  1.0,
                "raw_logit":      0.0,
                "coverage_a_to_b": 0.0,
                "coverage_b_to_a": 0.0,
            })
        else:
            to_score.append((i, cand))
            results.append(None)   # placeholder

    if not to_score:
        return results

    # Tokenise all pairs at once — single batch
    texts_b = [_strip_contextual_header(cand["text"]) for _, cand in to_score]
    texts_a  = [text_a_clean] * len(texts_b)

    inputs = tokenizer(
        texts_a, texts_b,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,   # pad to the same length within the batch
    ).to(DEVICE)

    outputs = None
    with _autocast_ctx():
        outputs = model(**inputs)

    logits_batch    = outputs.logits          # (batch, num_labels)
    attentions_last = outputs.attentions[-1]  # (batch, num_heads, seq, seq)

    for batch_i, (orig_i, cand) in enumerate(to_score):
        logits = logits_batch[batch_i]
        if logits.dim() == 0 or logits.shape[0] == 1:
            prob_dup  = float(torch.sigmoid(logits.squeeze()).item())
            raw_logit = logits.squeeze().item()
        else:
            prob_dup  = float(torch.softmax(logits, dim=-1)[-1].item())
            raw_logit = logits[-1].item()

        avg_attn = attentions_last[batch_i].mean(dim=0)  # (seq, seq)

        input_ids = inputs["input_ids"][batch_i]
        n_tokens  = int(inputs["attention_mask"][batch_i].sum().item())
        sep_id    = tokenizer.sep_token_id
        sep_pos   = (input_ids == sep_id).nonzero(as_tuple=True)[0]
        sep_idx   = int(sep_pos[0].item()) if len(sep_pos) > 0 else n_tokens // 2

        cov_a2b, cov_b2a = _max_alignment_coverage(avg_attn, sep_idx, n_tokens)
        nis = _novel_information_score(avg_attn, sep_idx, n_tokens)

        results[orig_i] = {
            **cand,
            "skipped":         False,
            "skip_reason":     "",
            "raw_logit":       round(raw_logit, 4),
            "prob_duplicate":  round(prob_dup, 4),
            "coverage_a_to_b": round(cov_a2b, 4),
            "coverage_b_to_a": round(cov_b2a, 4),
            "nis_b_given_a":   nis,
        }

    return results


@torch.no_grad()
def score_sentences_batched(candidate_text: str, sentences: list[str]) -> list[dict]:
    """
    Batch version of the per-sentence scoring used by sentence-level merge —
    tokenises one candidate against ALL of A's sentences together and runs a
    single forward pass, instead of one forward pass per sentence.

    This mirrors score_candidates_batched but with the roles reversed: there,
    one new chunk is fixed as text_a and many candidates vary as text_b; here,
    one candidate is fixed as text_a (playing the "known context" role for
    NIS, exactly as in the single-pair call this replaces:
    score_pair(candidate_text, sentence)) and A's sentences vary as text_b.
    Looping over the (few, same-document) candidates and batching over
    sentences inside each iteration turns m*k single-pair forward passes
    into k batched ones, with identical outputs (same tokenisation, same
    model, same math — only the batching changes).

    Input:
        candidate_text : text of one same-document candidate B_j (the "A"
                          role for NIS purposes, matching score_pair's usage
                          in the merge step: score_pair(cand["text"], sent)).
        sentences      : list of A's sentences (the "B" role), scored all
                          at once against candidate_text.

    Returns:
        list of dicts, one per sentence, in the same order as `sentences`,
        each with the same fields as score_pair (raw_logit, prob_duplicate,
        coverage_a_to_b, coverage_b_to_a, nis_b_given_a).
    """
    if not sentences:
        return []

    tokenizer, model = get_cross_encoder()

    texts_a = [candidate_text] * len(sentences)
    texts_b = list(sentences)

    inputs = tokenizer(
        texts_a, texts_b,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(DEVICE)

    with _autocast_ctx():
        outputs = model(**inputs)

    logits_batch    = outputs.logits          # (batch, num_labels)
    attentions_last = outputs.attentions[-1]  # (batch, num_heads, seq, seq)

    results = []
    for batch_i in range(len(sentences)):
        logits = logits_batch[batch_i]
        if logits.dim() == 0 or logits.shape[0] == 1:
            prob_dup  = float(torch.sigmoid(logits.squeeze()).item())
            raw_logit = logits.squeeze().item()
        else:
            prob_dup  = float(torch.softmax(logits, dim=-1)[-1].item())
            raw_logit = logits[-1].item()

        avg_attn = attentions_last[batch_i].mean(dim=0)  # (seq, seq)

        input_ids = inputs["input_ids"][batch_i]
        n_tokens  = int(inputs["attention_mask"][batch_i].sum().item())
        sep_id    = tokenizer.sep_token_id
        sep_pos   = (input_ids == sep_id).nonzero(as_tuple=True)[0]
        sep_idx   = int(sep_pos[0].item()) if len(sep_pos) > 0 else n_tokens // 2

        cov_a2b, cov_b2a = _max_alignment_coverage(avg_attn, sep_idx, n_tokens)
        nis = _novel_information_score(avg_attn, sep_idx, n_tokens)

        results.append({
            "raw_logit":        round(raw_logit, 4),
            "prob_duplicate":   round(prob_dup, 4),
            "coverage_a_to_b":  round(cov_a2b, 4),
            "coverage_b_to_a":  round(cov_b2a, 4),
            "nis_b_given_a":    nis,
        })

    return results


@torch.no_grad()
def score_pairs_batched(
    pairs: list[tuple[str, str]],
    sub_batch_size: int = 128,
) -> list[dict]:
    """
    Fully general batched scoring: unlike score_candidates_batched (one A,
    many B) or score_sentences_batched (one A per call, many B), here BOTH
    text_a and text_b vary independently per pair. This is what lets Stage 2
    batch across MULTIPLE new chunks at once (each chunk = a different A,
    each with its own K candidates = different B's), instead of one forward
    pass per chunk.

    Internally chunks the pair list into sub-batches of `sub_batch_size` to
    bound peak memory (attention tensors are O(batch x heads x seq x seq));
    each sub-batch is still a single forward pass, so this is still k
    forward passes total for k = ceil(len(pairs)/sub_batch_size), not one
    pair at a time.

    Callers are responsible for any guard filtering (parent-child skip,
    header stripping) before building `pairs` — this function does no
    guarding of its own, matching score_sentences_batched's contract.

    Input:
        pairs          : list of (text_a, text_b) tuples.
        sub_batch_size : max pairs per forward pass (memory safety valve).

    Returns:
        list of dicts, one per pair, in the same order as `pairs`, each
        with the same fields as score_pair (raw_logit, prob_duplicate,
        coverage_a_to_b, coverage_b_to_a, nis_b_given_a).
    """
    if not pairs:
        return []

    tokenizer, model = get_cross_encoder()
    results: list[dict] = []

    for start in range(0, len(pairs), sub_batch_size):
        sub = pairs[start:start + sub_batch_size]
        texts_a = [p[0] for p in sub]
        texts_b = [p[1] for p in sub]

        inputs = tokenizer(
            texts_a, texts_b,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(DEVICE)

        with _autocast_ctx():
            outputs = model(**inputs)

        logits_batch    = outputs.logits
        attentions_last  = outputs.attentions[-1]

        for batch_i in range(len(sub)):
            logits = logits_batch[batch_i]
            if logits.dim() == 0 or logits.shape[0] == 1:
                prob_dup  = float(torch.sigmoid(logits.squeeze()).item())
                raw_logit = logits.squeeze().item()
            else:
                prob_dup  = float(torch.softmax(logits, dim=-1)[-1].item())
                raw_logit = logits[-1].item()

            avg_attn = attentions_last[batch_i].mean(dim=0)

            input_ids = inputs["input_ids"][batch_i]
            n_tokens  = int(inputs["attention_mask"][batch_i].sum().item())
            sep_id    = tokenizer.sep_token_id
            sep_pos   = (input_ids == sep_id).nonzero(as_tuple=True)[0]
            sep_idx   = int(sep_pos[0].item()) if len(sep_pos) > 0 else n_tokens // 2

            cov_a2b, cov_b2a = _max_alignment_coverage(avg_attn, sep_idx, n_tokens)
            nis = _novel_information_score(avg_attn, sep_idx, n_tokens)

            results.append({
                "raw_logit":       round(raw_logit, 4),
                "prob_duplicate":  round(prob_dup, 4),
                "coverage_a_to_b": round(cov_a2b, 4),
                "coverage_b_to_a": round(cov_b2a, 4),
                "nis_b_given_a":   nis,
            })

    return results


def score_candidates(
    chunk_text: str,
    candidates: list[dict],
    chunk_parent_id: str | None = None,
    chunk_level: str | None = None,
) -> list[dict]:
    """
    Wrapper around score_candidates_batched.
    Preserves the original interface so stage3_decision.py requires no changes.
    """
    return score_candidates_batched(
        chunk_text, candidates, chunk_parent_id, chunk_level,
    )


import re as _re


def _strip_contextual_header(text: str) -> str:
    """
    Remove the [Context: ... ] header prepended by the Contextual chunker.

    Example:
      "[Context: Super Bowl 50 | Part 3/12] The game was..." => "The game was..."

    If no header is present, returns the text unchanged.
    The original text stored in Qdrant is NOT modified — stripping happens
    only before scoring to prevent false redundancy from identical headers.
    """
    return _re.sub(r'^\[Context:[^\]]*\]\s*', '', text).strip()


def _is_parent_child_pair(
    chunk_parent_id:  str | None,
    chunk_level:      str | None,
    cand_parent_id:   str | None,
    cand_level:       str | None,
    cand_chunk_id:    str,
) -> bool:
    """
    Return True if the new chunk and the candidate form a parent-child pair
    in HierarchicalParentChild chunking.

    Parent-child pairs overlap by design, not because of genuine content
    duplication, and must not be compared by the dedup scorer.

    Cases that trigger a skip:
      1. New chunk is a child whose parent is the candidate:
         chunk_parent_id == cand_chunk_id
      2. New chunk is a parent and the candidate is one of its children:
         chunk_level == "parent" and cand_parent_id is set
      3. Both are sibling children of the same parent:
         => NOT skipped; siblings may genuinely duplicate each other
            (small chunk_size + overlap) and should be processed normally.
    """
    # Case 1: new chunk is a child; candidate is its parent
    if chunk_parent_id and chunk_parent_id == cand_chunk_id:
        return True

    # Case 2: new chunk is a parent; candidate is one of its children
    if chunk_level == "parent" and cand_parent_id:
        if cand_parent_id.startswith(chunk_parent_id or "___NOMATCH___"):
            return True

    return False