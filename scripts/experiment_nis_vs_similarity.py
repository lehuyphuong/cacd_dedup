"""
scripts/experiment_nis_vs_similarity.py

Standalone experiment: compares CACD's New Information Score (NIS) against
plain cosine similarity (the Similarity baseline's signal) across a
controlled gradient of SEMANTIC overlap, from 100% down to 50% in 5% steps.

Why semantic overlap, not verbatim overlap: the central claim this paper
makes about pooled-vector similarity is that it can be misled once wording
changes, even when meaning does not. A verbatim-overlap test (sharing exact
sentences) does not probe that claim; a paraphrase-overlap test does. Chunk
B here is built from genuine paraphrases of chunk A's content, not copies
of it, so at 100% "overlap" the two chunks share no guaranteed literal
wording at all, only meaning.

How overlap is controlled: chunk A is always the same 10-sentence passage.
Chunk B keeps a paraphrased version of the first N sentences of chunk A
(same meaning, different wording and sentence structure) and replaces the
remaining (20 - N) sentences with sentences from an unrelated passage, so
chunk A and chunk B share exactly N/20 = the target overlap percentage of
their content by construction, not by estimation.

Sentence length: two earlier versions of this script were too long and
silently exceeded the tokenizer's MAX_LENGTH, truncating away the
differentiating content near the end of chunk_b for most overlap levels.
The first fix (raising MAX_LENGTH to 512) was not enough, because the
initial word-to-token ratio estimate (~1.3 tokens/word) was too low --
this content's numbers ("10,100", "1889") and proper nouns ("Gustave
Eiffel") fragment into more subword tokens than ordinary prose. Sentences
here (~7-8 words each) are sized against the ratio actually measured from
that run (~1.44 tokens/word), with margin. This script also asserts at
runtime that no truncation occurs (see _check_no_truncation) so that
failure mode is reported with the true token count instead of recurring
silently -- do not trust any result printed alongside a truncation
warning.

No dependency on the benchmark pipeline -- only requires:
  pip install transformers sentence-transformers torch numpy

Run from the project root:
  python scripts/experiment_nis_vs_similarity.py
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

CROSS_ENCODER_MODEL = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"
EMBED_MODEL         = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"

# Max sequence length for both the cross-encoder pair and the bi-encoder.
# 512 is the hard architectural ceiling for both models here (standard
# BERT-family position embeddings) -- it cannot be raised further; the
# fix for exceeding it is to shorten the input text, not raise this value.
MAX_LENGTH = 512

# Overlap levels to test: 100%, 90%, 80%, ..., 50% (6 levels, 10% steps).
# Only 10 sentence units are used in this version (see BASE_FACTS below),
# so 5% steps are not available; 10% is the finest granularity that keeps
# each sentence long enough to carry real, paraphrasable meaning while
# keeping the whole chunk under LENGTH_GUARD (see below).
OVERLAP_LEVELS = list(range(100, 45, -10))

# CACD's real length-aware guard (Section III-D / stage3_decision.py):
# a chunk longer than this is protected from being dropped unless its NIS
# falls below NIS_FLOOR. The chunks in the earlier version of this script
# (~800-950 characters) were always longer than this, so CACD's real
# decision logic would have kept every pair regardless of overlap level --
# the guard, not NIS or cosine_sim, was deciding the outcome. Chunks here
# are deliberately built to stay under this threshold so that comparison
# no longer applies.
LENGTH_GUARD = 300

# Similarity baseline threshold used elsewhere in this paper's experiments,
# shown here only to mark where cosine similarity would call two chunks
# duplicates at each overlap level.
SIMILARITY_THRESHOLD = 0.8

# ── Controlled-overlap chunk pairs (semantic, not verbatim) ────────────────

# Chunk A is always the concatenation of all 20 sentences below.
# Chunk A is always the concatenation of all 10 sentences below. Kept short
# (~230 characters total) so the whole chunk stays under LENGTH_GUARD; see
# _check_length_guard below, which verifies this at runtime for every pair.
BASE_FACTS = [
    "The tower is in Paris.",
    "It was built by Eiffel.",
    "Work began in 1887.",
    "It opened in 1889.",
    "It stands 330 meters tall.",
    "It was tallest until 1930.",
    "It is made of iron.",
    "It weighs 10,000 tons.",
    "Millions visit each year.",
    "It symbolizes France.",
]

# Genuine paraphrases of BASE_FACTS, same order, same meaning, deliberately
# different wording -- this is the "semantic overlap" content used to
# build chunk B, never a copy of BASE_FACTS.
PARAPHRASED_FACTS = [
    "The tower sits in Paris.",
    "Eiffel built the tower.",
    "Building started in 1887.",
    "It welcomed guests in 1889.",
    "It rises about 330 meters.",
    "It led in height until 1930.",
    "Iron makes up its frame.",
    "Its weight is near 10,000 tons.",
    "Many tourists come yearly.",
    "It stands for France.",
]

# Unrelated sentences used to replace PARAPHRASED_FACTS sentences in chunk B
# as the target overlap decreases.
DISTRACTOR_FACTS = [
    "The Amazon is in Brazil.",
    "It reaches into Peru too.",
    "It covers millions of km2.",
    "It's the largest rainforest.",
    "The Amazon River flows there.",
    "It holds many species.",
    "Communities live within it.",
    "It affects global climate.",
    "Deforestation harms the forest.",
    "It's called Earth's lungs.",
]

assert len(BASE_FACTS) == len(PARAPHRASED_FACTS) == len(DISTRACTOR_FACTS) == 10, \
    "All three fact lists must have exactly 10 sentences for clean 10% steps."


def build_pair(overlap_pct: int) -> tuple[str, str]:
    """
    Build one (chunk_a, chunk_b) pair at a target semantic-overlap level.

    chunk_a is always the full 10-sentence base passage. chunk_b keeps a
    paraphrase (never a copy) of the first n_keep sentences of chunk_a and
    replaces the rest with sentences from an unrelated passage, so the two
    chunks share exactly overlap_pct percent of their meaning by
    construction, with no guaranteed literal wording in common anywhere.
    """
    n_total = len(BASE_FACTS)
    n_keep  = round(n_total * overlap_pct / 100)
    chunk_a = " ".join(BASE_FACTS)
    chunk_b = " ".join(PARAPHRASED_FACTS[:n_keep] + DISTRACTOR_FACTS[n_keep:])
    return chunk_a, chunk_b


# ── Load models ──────────────────────────────────────────────────────────────

print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
cross_encoder = AutoModelForSequenceClassification.from_pretrained(
    CROSS_ENCODER_MODEL, output_attentions=True
)
cross_encoder.to(DEVICE)
cross_encoder.eval()

print(f"Loading bi-encoder: {EMBED_MODEL} on {DEVICE} ...")
bi_encoder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
bi_encoder.max_seq_length = MAX_LENGTH
print("Models loaded.\n")


def _check_no_truncation(chunk_a: str, chunk_b: str, overlap_pct: int) -> None:
    """
    Warn loudly if MAX_LENGTH is too small for this pair, instead of
    letting the tokenizer silently drop the tail of chunk_b -- exactly the
    bug that made an earlier version of this script report an identical
    NIS and cosine_sim across most overlap levels (the differentiating
    content near the end of chunk_b never reached the model).
    """
    true_len = len(tokenizer(chunk_a, chunk_b, truncation=False)["input_ids"])
    if true_len > MAX_LENGTH:
        print(
            f"  [WARNING] overlap={overlap_pct}%: pair is {true_len} tokens, "
            f"exceeds MAX_LENGTH={MAX_LENGTH}. The tail of chunk_b will be "
            f"truncated away -- results at this overlap level cannot be "
            f"trusted. Shorten the fact sentences or raise MAX_LENGTH."
        )


def _check_length_guard(chunk_a: str, chunk_b: str, overlap_pct: int) -> None:
    """
    Warn if either chunk exceeds CACD's real LENGTH_GUARD. A chunk longer
    than LENGTH_GUARD is protected from being dropped in the real pipeline
    unless its NIS falls below NIS_FLOOR (0.3); if that never happens on
    this gradient, the length guard -- not NIS or cosine_sim -- would be
    the thing actually deciding CACD's outcome for every pair, making the
    NIS-vs-cosine_sim comparison here moot for the real decision rule.
    """
    for name, chunk in [("chunk_a", chunk_a), ("chunk_b", chunk_b)]:
        if len(chunk) > LENGTH_GUARD:
            print(
                f"  [WARNING] overlap={overlap_pct}%: {name} is {len(chunk)} "
                f"characters, exceeds LENGTH_GUARD={LENGTH_GUARD}. In the "
                f"real CACD pipeline this pair would be protected from "
                f"being dropped whenever NIS > NIS_FLOOR (0.3), regardless "
                f"of what cosine_sim or NIS actually says."
            )


# ── Scoring functions ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_nis(chunk_a: str, chunk_b: str) -> dict:
    """
    Score (chunk_a, chunk_b) with the cross-encoder and compute the New
    Information Score exactly as defined in CACD: entropy of the B=>A
    attention (how much of chunk_b's tokens are explained by chunk_a),
    normalized by log|A|, averaged over the tokens of chunk_b.

    Returns dict with prob_dup (cross-encoder duplicate probability) and
    nis (New Information Score, in [0, 1]).
    """
    inputs = tokenizer(
        chunk_a, chunk_b,
        return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True,
    ).to(DEVICE)

    outputs = cross_encoder(**inputs)
    logits  = outputs.logits.squeeze()
    prob_dup = (
        float(torch.sigmoid(logits).item())
        if logits.dim() == 0
        else float(torch.softmax(logits, dim=-1)[-1].item())
    )

    last_attn = outputs.attentions[-1][0]      # (num_heads, seq, seq)
    avg_attn  = last_attn.mean(dim=0)          # (seq, seq)

    input_ids = inputs["input_ids"][0]
    n_tokens  = int(inputs["attention_mask"][0].sum().item())
    sep_id    = tokenizer.sep_token_id
    sep_pos   = (input_ids == sep_id).nonzero(as_tuple=True)[0]
    sep_idx   = int(sep_pos[0].item()) if len(sep_pos) > 0 else n_tokens // 2

    len_a   = sep_idx - 1
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)
    sub_b2a = avg_attn[b_range, a_range]       # tokens of B attending to A

    if sub_b2a.numel() == 0 or len_a < 2:
        nis = 1.0
    else:
        row_sums = sub_b2a.sum(dim=1, keepdim=True).clamp(min=1e-9)
        p_b2a    = sub_b2a / row_sums
        ent      = -(p_b2a * torch.log(p_b2a + 1e-9)).sum(dim=1)
        max_ent  = float(np.log(len_a))
        nis = (
            float((ent / max_ent).mean().clamp(0.0, 1.0).item())
            if max_ent > 1e-9 else 1.0
        )

    return {"prob_dup": round(prob_dup, 4), "nis": round(nis, 4)}


def compute_cosine_similarity(chunk_a: str, chunk_b: str) -> float:
    """Cosine similarity between chunk_a and chunk_b under the same
    bi-encoder used by the Similarity baseline (L2-normalized dense
    embeddings, dot product = cosine similarity)."""
    vecs = bi_encoder.encode([chunk_a, chunk_b], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 84)
    print("NIS vs. Cosine Similarity across a controlled semantic-overlap gradient")
    print(f"Cross-encoder: {CROSS_ENCODER_MODEL}")
    print(f"Bi-encoder:    {EMBED_MODEL}")
    print("=" * 84)
    print()

    rows = []
    for overlap_pct in OVERLAP_LEVELS:
        chunk_a, chunk_b = build_pair(overlap_pct)
        _check_no_truncation(chunk_a, chunk_b, overlap_pct)
        _check_length_guard(chunk_a, chunk_b, overlap_pct)
        cos_sim = compute_cosine_similarity(chunk_a, chunk_b)
        cacd    = compute_nis(chunk_a, chunk_b)

        rows.append({
            "overlap_pct": overlap_pct,
            "cosine_sim":  round(cos_sim, 4),
            "nis":         cacd["nis"],
            "prob_dup":    cacd["prob_dup"],
        })

        print(
            f"[{overlap_pct:3d}% overlap] "
            f"cosine_sim={cos_sim:.4f}  NIS={cacd['nis']:.4f}  "
            f"prob_dup={cacd['prob_dup']:.4f}"
        )

    # ── Summary table ────────────────────────────────────────────────────────
    print()
    print("=" * 84)
    print("SUMMARY TABLE")
    print("=" * 84)
    header = (
        f"{'Overlap %':>10} | {'Cosine Sim':>11} | {'Sim >= 0.8?':>12} | "
        f"{'NIS':>7} | {'1 - NIS':>8} | {'p_dup':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sim_flag = "DUP" if r["cosine_sim"] >= SIMILARITY_THRESHOLD else "keep"
        print(
            f"{r['overlap_pct']:>9}% | {r['cosine_sim']:>11.4f} | "
            f"{sim_flag:>12} | {r['nis']:>7.4f} | "
            f"{1 - r['nis']:>8.4f} | {r['prob_dup']:>7.4f}"
        )

    # ── Correlation with ground-truth overlap ──────────────────────────────
    overlaps      = np.array([r["overlap_pct"] for r in rows], dtype=float)
    cos_vals      = np.array([r["cosine_sim"] for r in rows])
    one_minus_nis = 1.0 - np.array([r["nis"] for r in rows])

    r_cos = np.corrcoef(overlaps, cos_vals)[0, 1]
    r_nis = np.corrcoef(overlaps, one_minus_nis)[0, 1]

    print()
    print("=" * 84)
    print("Correlation with the true overlap percentage (Pearson r; higher is better)")
    print("=" * 84)
    print(f"  cosine_sim  vs. overlap_pct : r = {r_cos:.4f}")
    print(f"  (1 - NIS)   vs. overlap_pct : r = {r_nis:.4f}")
    print()
    print("Interpretation:")
    print("  chunk_b is built from paraphrases at every overlap level, never")
    print("  literal copies, so any lexical overlap between chunk_a and")
    print("  chunk_b is incidental, not by design. This is the case a pooled")
    print("  bi-encoder vector is most at risk of misjudging: two chunks can")
    print("  share little surface wording while still meaning the same thing.")
    print("  Both cosine_sim and (1 - NIS) are read here as redundancy")
    print("  signals on a comparable [0, 1] scale: higher means the two")
    print("  chunks are judged more alike. Since overlap_pct is the exact,")
    print("  known ground truth by construction, the signal whose values")
    print("  track overlap_pct more closely (larger Pearson r, and a wider")
    print("  spread of scores across the 50-100% range instead of staying")
    print("  near a constant value) is the more informative redundancy")
    print("  signal on this gradient.")
    print("  'Sim >= 0.8?' marks where the Similarity baseline's fixed")
    print("  threshold would call the pair a duplicate at each overlap")
    print("  level; a threshold that stays DUP across most of the 50-100%")
    print("  range would fail to separate near-duplicate chunks from only")
    print("  loosely related ones.")


if __name__ == "__main__":
    main()