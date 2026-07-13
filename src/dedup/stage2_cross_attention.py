"""
CACD Stage 2 — Cross-Attention Redundancy Scorer.

Role: for each pair (new_chunk, candidate), produce a redundancy signal by
jointly encoding both texts through a cross-encoder and extracting
information from the resulting attention matrix.

Pipeline:
  1. Joint encoding: [CLS] new_chunk [SEP] candidate [SEP]
  2. Attention extraction: last transformer layer, averaged across heads
  3. Coverage signals: max-alignment coverage (BERTScore-style)
  4. Novel Information Score (NIS): entropy of attention candidate=>new_chunk

Model: cross-encoder/msmarco-MiniLM-L6-en-de-v1 (pretrained, no fine-tuning).

Attention capture: only the model's final transformer layer is forced into
eager attention mode (see _enable_last_layer_only_eager below); every other
layer keeps the faster default attention implementation. This is
functionally identical to requesting attention output from the whole model,
just faster, since NIS only ever reads the final layer.

Public API:
  get_cross_encoder()                    -> (tokenizer, model), lazy-loaded
  score_pair(text_a, text_b)             -> dict of scores for one pair
  score_candidates(...)                  -> list of scored candidate dicts
  score_candidates_batched(...)          -> batched version of the above
  score_pairs_batched(pairs)             -> scores an arbitrary list of
                                             (text_a, text_b) pairs in one
                                             or more batched forward passes
"""

from __future__ import annotations

import contextlib
import copy
import logging

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.settings import (
    CACD_CROSS_ENCODER_MODEL,
    CACD_USE_FP16,
    CACD_USE_LAST_LAYER_EAGER_ATTENTION,
    DEVICE,
)

logger = logging.getLogger(__name__)

_tokenizer = None
_model     = None

# True once get_cross_encoder() has confirmed the last-layer-only-eager patch
# is active and self-tested. False means the model is running the slower
# global-eager fallback, either because CACD_USE_LAST_LAYER_EAGER_ATTENTION
# is off or because the self-test failed on this model/transformers version.
_last_layer_capture_active = False

# Mutable box the forward hook writes captured attention weights into.
# A dict (not a plain variable) so the hook closure can mutate it without
# `nonlocal`/`global`. Forward passes run sequentially in this codebase, so
# one shared slot, overwritten and read back within the same call, is safe.
_captured_attn: dict = {"weights": None}


def _autocast_ctx():
    """Mixed-precision context for the cross-encoder forward pass. Enabled
    only on CUDA; a no-op on CPU regardless of CACD_USE_FP16."""
    if CACD_USE_FP16 and DEVICE == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _find_last_self_attention_module(model):
    """
    Locate the final transformer layer's self-attention submodule (e.g.
    model.bert.encoder.layer[-1].attention.self for a BERT-family
    sequence-classification model).

    Returns the submodule, or None if the model does not match the expected
    BERT-style structure.
    """
    try:
        base = getattr(model, model.base_model_prefix)
        last_layer = base.encoder.layer[-1]
        return last_layer.attention.self
    except (AttributeError, IndexError, TypeError):
        return None


def _enable_last_layer_only_eager(model) -> bool:
    """
    Make only the final transformer layer compute and expose real attention
    weights, while every other layer keeps the fast default attention path.

    How it works:
      1. The last layer's self-attention submodule gets its own private
         copy of the model config with `_attn_implementation` set to
         "eager". A copy is required because HuggingFace shares one config
         object across every layer by reference; mutating it in place
         would force the whole model into eager mode.
      2. A forward hook on that submodule captures its returned attention
         weights directly, independent of whether wrapping modules
         propagate them any further up the call stack.

    Returns True if the patch was applied; False if this model's
    architecture did not match what was expected, in which case the caller
    falls back to loading the model with global eager attention.
    """
    self_attn = _find_last_self_attention_module(model)
    if self_attn is None:
        logger.warning(
            "Last-layer-eager patch: could not locate a BERT-style "
            "encoder.layer[-1].attention.self submodule on %s; "
            "falling back to global eager loading.",
            type(model).__name__,
        )
        return False

    private_config = copy.copy(self_attn.config)
    private_config._attn_implementation = "eager"
    self_attn.config = private_config

    def _capture_hook(_module, _args, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            _captured_attn["weights"] = output[1]

    self_attn.register_forward_hook(_capture_hook)
    return True


def _self_test_last_layer_capture(tokenizer, model) -> bool:
    """
    Run one small forward pass to confirm the capture hook produces a
    real, correctly-shaped attention tensor before trusting it for a full
    benchmark run.

    Returns True if a non-None tensor of shape
    (batch, num_heads, seq_len, seq_len) was captured; False otherwise.
    """
    _captured_attn["weights"] = None
    probe = tokenizer(
        "self test", "probe pair",
        return_tensors="pt", truncation=True, max_length=16, padding=True,
    ).to(DEVICE)
    with torch.no_grad():
        model(**probe)

    weights = _captured_attn["weights"]
    _captured_attn["weights"] = None

    if weights is None:
        logger.warning("Last-layer-eager self-test: no attention weights captured.")
        return False
    if weights.dim() != 4 or weights.shape[0] != 1:
        logger.warning(
            "Last-layer-eager self-test: unexpected attention shape %s.",
            tuple(weights.shape),
        )
        return False
    return True


def _load_model_global_eager():
    """Load the model with attention output enabled on every layer.
    Used as the fallback when the last-layer-only patch is not applicable."""
    tokenizer = AutoTokenizer.from_pretrained(CACD_CROSS_ENCODER_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        CACD_CROSS_ENCODER_MODEL,
        output_attentions=True,
    )
    return tokenizer, model


def get_cross_encoder():
    """Lazy-load the cross-encoder model and tokenizer (pretrained, no
    fine-tuning). Returns (tokenizer, model)."""
    global _tokenizer, _model, _last_layer_capture_active
    if _model is not None:
        return _tokenizer, _model

    logger.info("Loading cross-encoder: %s", CACD_CROSS_ENCODER_MODEL)

    if not CACD_USE_LAST_LAYER_EAGER_ATTENTION:
        _tokenizer, _model = _load_model_global_eager()
        _last_layer_capture_active = False
        logger.info("Cross-encoder loaded on %s (global eager, flag disabled)", DEVICE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(CACD_CROSS_ENCODER_MODEL)
        model     = AutoModelForSequenceClassification.from_pretrained(
            CACD_CROSS_ENCODER_MODEL,
        )
        model.to(DEVICE)
        model.eval()

        patched = _enable_last_layer_only_eager(model)
        verified = patched and _self_test_last_layer_capture(tokenizer, model)

        if verified:
            _tokenizer, _model = tokenizer, model
            _last_layer_capture_active = True
            logger.info(
                "Cross-encoder loaded on %s (last-layer-only eager, "
                "self-test passed)", DEVICE,
            )
        else:
            logger.warning(
                "Last-layer-only eager patch failed self-test; "
                "falling back to global eager loading."
            )
            _tokenizer, _model = _load_model_global_eager()
            _model.to(DEVICE)
            _model.eval()
            _last_layer_capture_active = False
            logger.info("Cross-encoder loaded on %s (global eager, fallback)", DEVICE)

    return _tokenizer, _model


def _last_layer_attention(outputs, batch_i: int | None = None) -> torch.Tensor:
    """
    Retrieve the final layer's attention matrix for the just-completed
    forward pass, regardless of which loading path is active.

    Args:
        outputs : the model's forward-pass output object.
        batch_i : if given, return a single (num_heads, seq, seq) tensor
                  for that batch index; if None, return the full
                  (batch, num_heads, seq, seq) tensor.
    """
    if _last_layer_capture_active:
        weights = _captured_attn["weights"]
        _captured_attn["weights"] = None
        if weights is None:
            raise RuntimeError(
                "Last-layer-eager capture is active but no attention "
                "weights were captured for this forward pass."
            )
    else:
        weights = outputs.attentions[-1]

    return weights if batch_i is None else weights[batch_i]


def _max_alignment_coverage(
    attn: torch.Tensor,
    sep_idx: int,
    n_tokens: int,
) -> tuple[float, float]:
    """
    Compute max-alignment coverage from the attention matrix (BERTScore-style).

    coverage(A=>B) = mean over tokens in A of the maximum attention weight
                     that token assigns to any token in B.
    coverage(B=>A) = same, in the opposite direction.

    Args:
        attn    : attention matrix (n_tokens, n_tokens), heads already averaged.
        sep_idx : position of the first [SEP] token (A/B boundary).
        n_tokens: total number of real tokens (excluding padding).

    Returns:
        (coverage_a_to_b, coverage_b_to_a)
    """
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)

    sub_a_to_b = attn[a_range, b_range]
    sub_b_to_a = attn[b_range, a_range]

    if sub_a_to_b.numel() == 0 or sub_b_to_a.numel() == 0:
        return 0.0, 0.0

    coverage_a_to_b = sub_a_to_b.max(dim=1).values.mean().item()
    coverage_b_to_a = sub_b_to_a.max(dim=1).values.mean().item()

    return coverage_a_to_b, coverage_b_to_a


def _novel_information_score(
    attn: torch.Tensor,
    sep_idx: int,
    n_tokens: int,
) -> float:
    """
    Novel Information Score (NIS): how much new information B carries
    relative to A, based on the entropy of the attention B=>A.

    Formula:
      1. Re-normalize the B=>A attention sub-matrix row-wise.
      2. H(j) = -sum_i p(i|j) * log(p(i|j)) per token j in B.
      3. NIS = mean(H(j)) / log(|A|), clipped to [0, 1].

    Args:
        attn    : attention matrix (n_tokens, n_tokens), heads already averaged.
        sep_idx : position of the first [SEP] token (A/B boundary).
        n_tokens: total number of real tokens.

    Returns:
        nis: float in [0, 1]. Near 0 means B is fully explained by A
             (redundant); near 1 means B is largely novel relative to A.
    """
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)

    sub_b_to_a = attn[b_range, a_range]

    if sub_b_to_a.numel() == 0:
        return 1.0

    len_a = sub_b_to_a.shape[1]
    if len_a < 2:
        return 1.0

    row_sums    = sub_b_to_a.sum(dim=1, keepdim=True).clamp(min=1e-9)
    prob_b_to_a = sub_b_to_a / row_sums

    ent_per_token = -(prob_b_to_a * torch.log(prob_b_to_a + 1e-9)).sum(dim=1)
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
        raw_logit       : float -- raw cross-encoder output
        prob_duplicate  : float -- sigmoid(logit) for binary models,
                                   softmax[-1] for multi-class models
        coverage_a_to_b : float -- fraction of A's content covered by B
        coverage_b_to_a : float -- fraction of B's content covered by A
        nis_b_given_a   : float -- Novel Information Score of B given A
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

    logits = outputs.logits.squeeze()
    if logits.dim() == 0:
        raw_logit = logits.item()
        prob_dup  = float(torch.sigmoid(logits).item())
    else:
        raw_logit = logits[-1].item()
        prob_dup  = float(torch.softmax(logits, dim=-1)[-1].item())

    last_layer_attn = _last_layer_attention(outputs, batch_i=0)
    avg_attn        = last_layer_attn.mean(dim=0)

    input_ids = inputs["input_ids"][0]
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
    }


@torch.no_grad()
def score_candidates_batched(
    chunk_text: str,
    candidates: list[dict],
    chunk_parent_id: str | None = None,
    chunk_level: str | None = None,
) -> list[dict]:
    """
    Batch-scores one new chunk against all of its candidates in a single
    forward pass instead of one pass per candidate.

    Input:
        chunk_text     : text of the new chunk (A).
        candidates     : list of candidate dicts from Stage 1.
        chunk_parent_id: parent_id of the new chunk (parent-child guard).
        chunk_level    : level field of the new chunk (parent-child guard).

    Returns:
        list of candidate dicts, each extended with scoring fields
        (prob_duplicate, nis_b_given_a, coverage_*, skipped, skip_reason).
    """
    tokenizer, model = get_cross_encoder()
    text_a_clean = _strip_contextual_header(chunk_text)

    to_score: list[tuple[int, dict]] = []
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
            results.append(None)

    if not to_score:
        return results

    texts_b = [_strip_contextual_header(cand["text"]) for _, cand in to_score]
    texts_a  = [text_a_clean] * len(texts_b)

    inputs = tokenizer(
        texts_a, texts_b,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(DEVICE)

    with _autocast_ctx():
        outputs = model(**inputs)

    logits_batch = outputs.logits
    attentions_last = _last_layer_attention(outputs)

    for batch_i, (orig_i, cand) in enumerate(to_score):
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
def score_pairs_batched(
    pairs: list[tuple[str, str]],
    sub_batch_size: int = 128,
) -> list[dict]:
    """
    General batched scoring where both text_a and text_b vary independently
    per pair. This lets Stage 3 batch across multiple new chunks at once
    (each with its own K candidates) instead of one forward pass per chunk.

    Splits `pairs` into sub-batches of `sub_batch_size` to bound peak
    memory. Callers are responsible for any guard filtering (parent-child
    skip, header stripping) before building `pairs`.

    Input:
        pairs          : list of (text_a, text_b) tuples.
        sub_batch_size : max pairs per forward pass.

    Returns:
        list of dicts, one per pair, in the same order as `pairs`, each
        with the same fields as score_pair.
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

        logits_batch = outputs.logits
        attentions_last = _last_layer_attention(outputs)

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
    """Thin wrapper around score_candidates_batched, kept for interface
    compatibility with callers that predate the batched implementation."""
    return score_candidates_batched(
        chunk_text, candidates, chunk_parent_id, chunk_level,
    )


import re as _re


def _strip_contextual_header(text: str) -> str:
    """
    Remove a "[Context: ...]" header prepended by the Contextual chunker,
    e.g. "[Context: Super Bowl 50 | Part 3/12] The game was..." becomes
    "The game was...". Returns the text unchanged if no header is present.
    Stripping happens only before scoring; the stored text is untouched.
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
    under HierarchicalParentChild chunking, in which case they must be
    excluded from scoring (they overlap by design, not by duplication).

    Sibling children of the same parent are NOT excluded; they may
    genuinely duplicate each other and are scored normally.
    """
    if chunk_parent_id and chunk_parent_id == cand_chunk_id:
        return True

    if chunk_level == "parent" and cand_parent_id:
        if cand_parent_id.startswith(chunk_parent_id or "___NOMATCH___"):
            return True

    return False
