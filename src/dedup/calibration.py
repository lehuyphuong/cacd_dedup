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

Note on RunningLogitCalibrator (retained for reference):
  The z-score online calibrator was the original Stage 3 signal before NIS was
  introduced. It is no longer used in the active decision path but is kept here
  for comparison experiments.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RunningLogitCalibrator:
    """
    Online (streaming) calibrator that converts a raw redundancy signal into
    a relative probability P(duplicate) using z-score normalisation + sigmoid.

    Input : redundancy_signal = min(coverage_a_to_b, coverage_b_to_a)
    Output: float in [0, 1] — probability that the pair is a duplicate,
            relative to the distribution of signals seen so far.

    Formula:
        z     = (signal - mean(observed)) / std(observed)
        P_dup = sigmoid(z) = 1 / (1 + exp(-z))

    This replaces a fixed absolute threshold with a distribution-relative
    comparison: a pair is flagged only when its signal is in the high tail
    of the observed distribution, regardless of domain-level scale shifts.
    """

    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples
        self._signals: list[float] = []

    def update(self, signal: float) -> None:
        self._signals.append(signal)

    def calibrated_probability(self, signal: float) -> float:
        """
        P(duplicate) = sigmoid(z-score(signal)).

        Returns 0.0 when fewer than 2 samples have been observed
        (neutral, biased toward KEEP to avoid premature drops).
        """
        if len(self._signals) < 2:
            return 0.0

        arr  = np.array(self._signals)
        mean = arr.mean()
        std  = arr.std() + 1e-6
        z    = (signal - mean) / std
        return float(1.0 / (1.0 + np.exp(-z)))

    def stats(self) -> dict:
        if not self._signals:
            return {"n": 0, "mean": 0.0, "std": 0.0}
        arr = np.array(self._signals)
        return {
            "n":    len(arr),
            "mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std()), 4),
        }


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
