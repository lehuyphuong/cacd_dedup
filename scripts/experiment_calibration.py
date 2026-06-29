"""
scripts/experiment_calibration.py

Standalone experiment: runs cross-attention scoring for 5 chunk pairs
with varying degrees of similarity, outputting heatmaps and decisions.

No dependency on the benchmark pipeline — only requires:
  pip install transformers torch matplotlib numpy

Run from the project root:
  python scripts/experiment_calibration.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR    = Path("results/calibration_experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Bayes-optimal cutoff (symmetric: cost_FP = cost_FN = 1.0)
BAYES_CUTOFF = 0.5

# ── 5 experimental chunk pairs ────────────────────────────────────────────────

PAIRS = {
    "case1_identical": {
        "label": "Case 1 — identical sentences (100% identical)",
        "a": "The Eiffel Tower was built in 1889 in Paris, France.",
        "b": "The Eiffel Tower was built in 1889 in Paris, France.",
    },
    "case2_paraphrase_full": {
        "label": "Case 2 — same meaning, different wording (100% paraphrase)",
        "a": "The Eiffel Tower was built in 1889 in Paris, France.",
        "b": "France's iconic Eiffel Tower was constructed in the year 1889 in the city of Paris.",
    },
    "case3_identical_50pct": {
        "label": "Case 3 — 50% identical (shared first half, different second half)",
        "a": "The Eiffel Tower was built in 1889 in Paris, France. It attracts millions of tourists every year.",
        "b": "The Eiffel Tower was built in 1889 in Paris, France. It was designed by the engineer Gustave Eiffel.",
    },
    "case4_paraphrase_50pct": {
        "label": "Case 4 — ~50% overlap with paraphrase + new information",
        "a": "The Eiffel Tower was built in 1889 in Paris, France. It attracts millions of tourists every year.",
        "b": "France's iconic tower was constructed in 1889 in Paris. It was designed by the engineer Gustave Eiffel.",
    },
    "case5_different": {
        "label": "Case 5 — completely different sentences",
        "a": "The Eiffel Tower was built in 1889 in Paris, France.",
        "b": "The Amazon rainforest covers over 5.5 million square kilometers in South America.",
    },
}

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading cross-encoder: {MODEL_NAME} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, output_attentions=True
)
model.to(DEVICE)
model.eval()
print("Model loaded.\n")

# ── Scoring functions ─────────────────────────────────────────────────────────

@torch.no_grad()
def score_pair(text_a: str, text_b: str) -> dict:
    inputs = tokenizer(
        text_a, text_b,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(DEVICE)

    outputs = model(**inputs)
    raw_logit = outputs.logits.squeeze().item()

    # Last attention layer, averaged across all heads
    last_attn  = outputs.attentions[-1][0]        # (num_heads, seq, seq)
    avg_attn   = last_attn.mean(dim=0)             # (seq, seq)

    input_ids  = inputs["input_ids"][0]
    tokens     = tokenizer.convert_ids_to_tokens(input_ids)
    n_tokens   = int(inputs["attention_mask"][0].sum().item())

    # First [SEP] position marks the A/B boundary
    sep_id        = tokenizer.sep_token_id
    sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)[0]
    sep_idx       = int(sep_positions[0].item()) if len(sep_positions) > 0 else n_tokens // 2

    # Sub-matrices A => B and B => A
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)
    sub_a2b = avg_attn[a_range, b_range]
    sub_b2a = avg_attn[b_range, a_range]

    # Max-alignment coverage (BERTScore-style)
    cov_a2b = sub_a2b.max(dim=1).values.mean().item() if sub_a2b.numel() > 0 else 0.0
    cov_b2a = sub_b2a.max(dim=1).values.mean().item() if sub_b2a.numel() > 0 else 0.0

    # Attention entropy (A => B)
    if sub_a2b.numel() > 0:
        probs    = sub_a2b / (sub_a2b.sum(dim=1, keepdim=True) + 1e-9)
        ent      = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
        max_ent  = math.log(max(sub_a2b.shape[1], 2))
        entropy  = float((ent / max_ent).mean().item())
    else:
        entropy  = 1.0

    redundancy_signal = min(cov_a2b, cov_b2a)

    return {
        "raw_logit":         round(raw_logit, 4),
        "coverage_a_to_b":   round(cov_a2b, 4),
        "coverage_b_to_a":   round(cov_b2a, 4),
        "redundancy_signal": round(redundancy_signal, 4),
        "attn_entropy":      round(entropy, 4),
        "attention_matrix":  avg_attn[:n_tokens, :n_tokens].cpu().numpy(),
        "tokens_a":          tokens[1:sep_idx],
        "tokens_b":          tokens[sep_idx + 1:n_tokens - 1],
        "sep_idx":           sep_idx,
        "n_tokens":          n_tokens,
    }


def decide(result: dict, cutoff: float = BAYES_CUTOFF) -> tuple[str, float]:
    """
    Decision based on redundancy_signal against a fixed absolute threshold.
    Used for visual inspection in this experiment only.

    Threshold ABS_CUTOFF = 0.05: based on observed audit-log distributions
    (all signals below 0.027 when no true duplicate exists).
    A principled threshold (GMM-BIC) is derived from the full corpus in
    calibration.py.
    """
    ABS_CUTOFF = 0.05
    signal = result["redundancy_signal"]
    if signal > ABS_CUTOFF:
        return "DROP", signal
    else:
        return "KEEP", signal


def plot_heatmap(result: dict, case_key: str, case_label: str, decision: str) -> str:
    attn     = result["attention_matrix"]
    sep_idx  = result["sep_idx"]
    n_tokens = result["n_tokens"]

    a_range  = slice(1, sep_idx)
    b_range  = slice(sep_idx + 1, n_tokens - 1)
    sub_attn = attn[a_range, b_range]

    tokens_a = result["tokens_a"]
    tokens_b = result["tokens_b"]

    if sub_attn.size == 0 or len(tokens_a) == 0 or len(tokens_b) == 0:
        print(f"  [WARNING] Empty attention sub-matrix for {case_key}, skipping heatmap.")
        return ""

    fig_w = max(6, min(0.45 * len(tokens_b), 22))
    fig_h = max(4, min(0.45 * len(tokens_a), 16))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(sub_attn, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=sub_attn.max())

    ax.set_xticks(range(len(tokens_b)))
    ax.set_xticklabels(tokens_b, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tokens_a)))
    ax.set_yticklabels(tokens_a, fontsize=7)
    ax.set_xlabel("Candidate (B)", fontsize=9)
    ax.set_ylabel("New chunk (A)", fontsize=9)

    title = (
        f"{case_label}\n"
        f"cov(A=>B)={result['coverage_a_to_b']:.4f}  "
        f"cov(B=>A)={result['coverage_b_to_a']:.4f}  "
        f"redundancy={result['redundancy_signal']:.4f}  "
        f"entropy={result['attn_entropy']:.3f}\n"
        f"raw_logit={result['raw_logit']:.3f}  "
        f"decision={decision}"
    )
    ax.set_title(title, fontsize=8, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention weight")
    fig.tight_layout()

    out_path = OUT_DIR / f"{case_key}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return str(out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_results = {}

    print("=" * 70)
    print("CACD — Calibration Experiment: 5 chunk pairs")
    print(f"Model: {MODEL_NAME} | Device: {DEVICE}")
    print(f"Output: {OUT_DIR.resolve()}")
    print("=" * 70)
    print()

    for case_key, pair in PAIRS.items():
        print(f"--- {pair['label']} ---")
        print(f"  A: {pair['a']}")
        print(f"  B: {pair['b']}")

        result = score_pair(pair["a"], pair["b"])
        decision, signal = decide(result)

        all_results[case_key] = {**result, "decision": decision}

        print(f"  coverage_a_to_b  : {result['coverage_a_to_b']:.4f}")
        print(f"  coverage_b_to_a  : {result['coverage_b_to_a']:.4f}")
        print(f"  redundancy_signal: {result['redundancy_signal']:.4f}  (= min of coverages)")
        print(f"  attn_entropy     : {result['attn_entropy']:.4f}")
        print(f"  raw_logit        : {result['raw_logit']:.4f}")
        print(f"  => DECISION      : {decision}")

        heatmap_path = plot_heatmap(result, case_key, pair["label"], decision)
        if heatmap_path:
            print(f"  => Heatmap saved : {heatmap_path}")
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Case':<35} {'cov(A=>B)':>10} {'cov(B=>A)':>10} {'redundancy':>12} {'decision':>8}")
    print("-" * 70)
    for case_key, r in all_results.items():
        print(
            f"{case_key:<35} "
            f"{r['coverage_a_to_b']:>10.4f} "
            f"{r['coverage_b_to_a']:>10.4f} "
            f"{r['redundancy_signal']:>12.4f} "
            f"{r['decision']:>8}"
        )

    print()
    print("Interpretation guide:")
    print("  redundancy_signal = min(cov_A=>B, cov_B=>A)")
    print("  HIGH (> 0.05) => DROP  (both directions show high coverage)")
    print("  LOW  (<= 0.05) => KEEP (at least one direction shows low coverage)")
    print("  Heatmap: dark red cell = token A attends strongly to token B")
    print("           diagonal pattern = tokens match in order => likely true duplicate")
    print("           single bright column = attention sink (model artefact, not duplication)")


if __name__ == "__main__":
    main()
