"""
Evaluation metrics — rag-bench v4.

Follows Berdyugina et al. (arXiv:2604.24334) Section 3.6.
Only the 4 core metrics are retained:

  1. Precision  = |Te ∩ Tr| / |Tr|
  2. Recall     = |Te ∩ Tr| / |Te|
  3. IoU        = |Te ∩ Tr| / |Te ∪ Tr|   (Jaccard)
  4. Index Size = chunk_count_after_filter + storage_mb  (reported separately)

where  Te = token set of reference passage
       Tr = token set of union of retrieved top-k chunks

Two tokenization modes (paper Section 3.6):
  raw          : lowercase word tokens only
  preprocessed : remove English stopwords + lemmatize (spaCy)

Oracle and generation metrics (EM / Token F1) are intentionally removed.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable


# ── Tokenizers ────────────────────────────────────────────────────────────────

def _raw_tokens(text: str) -> set[str]:
    """Lowercase word tokens — 'raw' mode."""
    return set(re.findall(r"\b\w+\b", text.lower()))


@lru_cache(maxsize=1)
def _get_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        raise RuntimeError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )


def _preprocessed_tokens(text: str) -> set[str]:
    """Lowercase, remove stopwords, lemmatize — 'preprocessed' mode."""
    nlp = _get_spacy()
    doc = nlp(text.lower())
    return {
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    }


def get_tokenizer(mode: str) -> Callable[[str], set[str]]:
    if mode == "raw":
        return _raw_tokens
    elif mode == "preprocessed":
        return _preprocessed_tokens
    else:
        raise ValueError(f"Unknown eval mode: {mode!r}. Choose 'raw' or 'preprocessed'.")


# ── Core retrieval metrics ────────────────────────────────────────────────────

def compute_retrieval_metrics(
    reference_text:   str,
    retrieved_chunks: list[dict],
    mode: str = "raw",
) -> dict[str, float]:
    """
    Compute Precision, Recall, IoU for one query.

    Args:
        reference_text  : ground-truth passage (SQuAD context).
        retrieved_chunks: list of chunk dicts with "text" field.
        mode            : "raw" or "preprocessed".

    Returns:
        {"precision": float, "recall": float, "iou": float}
    """
    tokenize = get_tokenizer(mode)

    te = tokenize(reference_text)
    if not te:
        return {"precision": 0.0, "recall": 0.0, "iou": 0.0}

    tr: set[str] = set()
    for chunk in retrieved_chunks:
        tr |= tokenize(chunk["text"])

    intersection = te & tr
    union        = te | tr

    precision = len(intersection) / len(tr)    if tr    else 0.0
    recall    = len(intersection) / len(te)
    iou       = len(intersection) / len(union) if union else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "iou":       round(iou,       4),
    }
