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

How overlap is controlled: chunk A is always the same 20-sentence passage.
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

# Overlap levels to test: 100%, 95%, 90%, ..., 50% (11 levels)
OVERLAP_LEVELS = list(range(100, 45, -5))

# Similarity baseline threshold used elsewhere in this paper's experiments,
# shown here only to mark where cosine similarity would call two chunks
# duplicates at each overlap level.
SIMILARITY_THRESHOLD = 0.8

# ── Controlled-overlap chunk pairs (semantic, not verbatim) ────────────────

# Chunk A is always the concatenation of all 20 sentences below.
BASE_FACTS = [
    "The Eiffel Tower stands in Paris, France.",
    "Gustave Eiffel designed and built the tower.",
    "Construction began in January 1887.",
    "The tower opened to the public in 1889.",
    "It was built for the 1889 World's Fair.",
    "The tower stands about 330 meters tall.",
    "It was the tallest structure until 1930.",
    "The tower is made of wrought iron.",
    "It weighs about ten thousand tons.",
    "The tower has three public levels.",
    "Seven million people visit it yearly.",
    "It is one of the world's most recognizable landmarks.",
    "The tower is repainted every seven years.",
    "Repainting uses about sixty tons of paint.",
    "Strong winds make the tower sway slightly.",
    "It was originally meant to be temporary.",
    "The tower has 108 stories architecturally.",
    "Its base forms a square 125 meters wide.",
    "Twenty thousand bulbs light it at night.",
    "It remains a lasting symbol of France.",
]

# Genuine paraphrases of BASE_FACTS, same order, same meaning, deliberately
# different wording and sentence structure -- this is the "semantic overlap"
# content used to build chunk B, never a copy of BASE_FACTS.
PARAPHRASED_FACTS = [
    "The famous tower is located in Paris.",
    "The landmark was designed by Gustave Eiffel.",
    "Building work started at the beginning of 1887.",
    "The tower welcomed visitors starting in 1889.",
    "It served as the gateway for the 1889 World's Fair.",
    "The structure reaches roughly 330 meters high.",
    "For decades, it was the tallest man-made structure.",
    "Wrought iron makes up the tower's frame.",
    "The completed tower weighs around 10,100 tons.",
    "Visitors can access three separate levels.",
    "About seven million tourists visit each year.",
    "Few landmarks are as widely recognized as this one.",
    "The tower gets repainted roughly every seven years.",
    "Each repainting takes close to sixty tons of paint.",
    "The upper tower sways a bit in high wind.",
    "Engineers originally planned it as a temporary structure.",
    "The frame is often described as 108 stories.",
    "The base is shaped like a square, 125 meters wide.",
    "About twenty thousand bulbs illuminate it nightly.",
    "It still stands today as a symbol of France.",
]

# Unrelated sentences used to replace PARAPHRASED_FACTS sentences in chunk B
# as the target overlap decreases, keeping chunk length roughly constant so
# overlap percentage is not confounded with chunk length.
DISTRACTOR_FACTS = [
    "The Amazon rainforest covers much of Brazil.",
    "It also extends into Peru and Colombia.",
    "The forest spans about 5.5 million square kilometers.",
    "It is the largest tropical rainforest on Earth.",
    "The Amazon River flows through the forest.",
    "It carries more water than any other river.",
    "Millions of species live within the forest.",
    "It holds roughly ten percent of known species.",
    "Indigenous communities have lived there for generations.",
    "The forest helps regulate the global climate.",
    "It produces about twenty percent of the world's oxygen.",
    "Large areas are lost to deforestation yearly.",
    "Logging and farming drive most forest loss.",
    "In places, the canopy exceeds forty meters.",
    "Some regions get over two thousand millimeters of rain.",
    "The forest hosts thousands of bird species.",
    "Jaguars, sloths, and river dolphins live there too.",
    "Scientists keep finding new species in the forest.",
    "Conservation groups work to protect the forest.",
    "The Amazon is often called the planet's lungs.",
]

assert len(BASE_FACTS) == len(PARAPHRASED_FACTS) == len(DISTRACTOR_FACTS) == 20, \
    "All three fact lists must have exactly 20 sentences for clean 5% steps."


def build_pair(overlap_pct: int) -> tuple[str, str]:
    """
    Build one (chunk_a, chunk_b) pair at a target semantic-overlap level.

    chunk_a is always the full 20-sentence base passage. chunk_b keeps a
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