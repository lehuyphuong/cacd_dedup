"""
CACD calibration — Bayes-optimal cutoff derivation.

Role: derive the DROP/KEEP threshold from a cost ratio rather than
a hand-picked constant (e.g. cosine threshold 0.8).

Formula (Elkan, 2001 — binary classification with asymmetric costs):
    cutoff = cost_FP / (cost_FP + cost_FN)

  cost_FP: cost of incorrectly dropping a non-duplicate chunk (information loss).
  cost_FN: cost of incorrectly keeping a duplicate chunk (wasted index space).

With symmetric costs (cost_FP = cost_FN = 1.0) => cutoff = 0.5, which coincides
with the natural sigmoid threshold of the cross-encoder model.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def bayes_optimal_cutoff(cost_false_positive: float, cost_false_negative: float) -> float:
    """
    Derive the decision cutoff from the cost ratio (Elkan, 2001).

    Args:
        cost_false_positive: cost of incorrectly dropping a non-duplicate
                             chunk (information loss).
        cost_false_negative: cost of incorrectly keeping a duplicate chunk
                             (wasted index space).

    Returns:
        cutoff in (0, 1).

    Formula:
        cutoff = cost_FP / (cost_FP + cost_FN)

    With equal costs => cutoff = 0.5 (symmetric default).
    Higher cost_FP => cutoff > 0.5 => system is more conservative when dropping.
    """
    denom = cost_false_positive + cost_false_negative
    if denom <= 0:
        return 0.5
    return cost_false_positive / denom
