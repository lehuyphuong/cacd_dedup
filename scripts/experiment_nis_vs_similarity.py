"""
scripts/experiment_nis_vs_similarity.py

Standalone experiment: compares CACD's New Information Score (NIS) against
plain cosine similarity (the Similarity baseline's signal) across a
controlled gradient of SEMANTIC overlap, from 100% down to 0% in 10% steps,
repeated across several independent topic pairs and averaged.

Why semantic overlap, not verbatim overlap: the central claim this paper
makes about pooled-vector similarity is that it can be misled once wording
changes, even when meaning does not. A verbatim-overlap test (sharing exact
sentences) does not probe that claim; a paraphrase-overlap test does. Chunk
B here is built from genuine paraphrases of chunk A's content, not copies
of it, so at 100% "overlap" the two chunks share no guaranteed literal
wording at all, only meaning.

Why several topic pairs, not one: a single hand-built example (e.g. Eiffel
Tower vs. Amazon rainforest) is an anecdote, not evidence -- a pattern seen
on one topic pair could easily be specific to that pair's vocabulary rather
than a general property of NIS or cosine similarity. TOPIC_PAIRS below
holds several independent (base, paraphrase, distractor) triples; results
are reported per pair and averaged across pairs.

How overlap is controlled: chunk A is always the same base passage for a
given topic pair. Chunk B keeps a paraphrase (never a copy) of the first
N sentences of chunk A and replaces the remaining sentences with sentences
from an unrelated passage, so chunk A and chunk B share exactly the target
overlap percentage of their meaning by construction, not by estimation.

Sentence and chunk length: kept short (~7-8 words per sentence, whole
chunk under ~300 characters) for two independent reasons found while
developing this script:
  1. Two earlier, longer-sentence versions silently exceeded the
     tokenizer's MAX_LENGTH, truncating away the differentiating content
     near the end of chunk_b for most overlap levels. This script asserts
     at runtime that no truncation occurs (see _check_no_truncation).
  2. Chunks longer than CACD's real LENGTH_GUARD (300 characters) are
     protected from being dropped whenever NIS > NIS_FLOOR (0.3) in the
     actual pipeline, which would make the length guard -- not NIS or
     cosine_sim -- the thing deciding the outcome. This script also
     asserts at runtime that no chunk exceeds LENGTH_GUARD (see
     _check_length_guard).
Do not trust any result printed alongside either warning.

No dependency on the benchmark pipeline -- only requires:
  pip install transformers sentence-transformers torch numpy scipy

Run from the project root:
  python scripts/experiment_nis_vs_similarity.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
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

# Overlap levels to test per topic pair: 100%, 90%, ..., 0% (11 levels,
# 10% steps). Extended down to 0% after an earlier run on the 100-50%
# range showed prob_dup pinned near 1.0 throughout that range -- 0% is
# needed to see whether prob_dup ever leaves CACD's confident-duplicate
# zone (>= PROB_HIGH = 0.8) for this kind of content.
OVERLAP_LEVELS = list(range(100, -5, -10))

# CACD's real length-aware guard (Section III-D / stage3_decision.py): a
# chunk longer than this is protected from being dropped unless its NIS
# falls below NIS_FLOOR. All chunks built below are kept under this
# threshold (checked at runtime) so that guard cannot mask the comparison.
LENGTH_GUARD = 300

# Similarity baseline threshold used elsewhere in this paper's experiments,
# shown here only to mark where cosine similarity would call two chunks
# duplicates at each overlap level.
SIMILARITY_THRESHOLD = 0.8

# ── Controlled-overlap topic pairs (semantic, not verbatim) ────────────────
#
# Each topic pair has three same-length sentence lists:
#   base        -- chunk_a's fixed content
#   paraphrase  -- same meaning as base, different wording; used to build
#                  the "kept" (overlapping) portion of chunk_b
#   distractor  -- unrelated content; used to build the "replaced"
#                  (non-overlapping) portion of chunk_b
# All three lists in a pair must be the same length (10 here, giving clean
# 10% steps); different pairs need not share the same length as each other.

TOPIC_PAIRS = [
    {
        "name": "Eiffel Tower / Amazon rainforest",
        "base": [
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
        ],
        "paraphrase": [
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
        ],
        "distractor": [
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
        ],
    },
    {
        "name": "Great Wall of China / Honeybee colonies",
        "base": [
            "The Great Wall is in China.",
            "It was built to stop invasions.",
            "Building began long ago.",
            "Dynasties added new sections.",
            "It spans thousands of kilometers.",
            "Sections use stone and earth.",
            "Watchtowers line the wall.",
            "Millions of workers built it.",
            "Tourists visit it yearly.",
            "It symbolizes China.",
        ],
        "paraphrase": [
            "The wall lies in China.",
            "It aimed to block invasions.",
            "Work started long ago.",
            "Later dynasties extended it.",
            "It runs thousands of kilometers.",
            "Stone and earth form sections.",
            "Towers stand along the wall.",
            "It took millions of laborers.",
            "Many tourists come each year.",
            "It represents China today.",
        ],
        "distractor": [
            "Bee colonies house many bees.",
            "One queen lays most eggs.",
            "Workers gather nectar and pollen.",
            "Bees dance to communicate.",
            "Hives keep steady temperatures.",
            "Bees make honey for energy.",
            "Colonies can last for years.",
            "Bees pollinate many crops.",
            "Pesticides threaten bee colonies.",
            "Beekeepers manage hives closely.",
        ],
    },
]

for _tp in TOPIC_PAIRS:
    assert len(_tp["base"]) == len(_tp["paraphrase"]) == len(_tp["distractor"]), \
        f"Topic pair '{_tp['name']}': base/paraphrase/distractor must be the same length."


def build_pair(topic: dict, overlap_pct: int) -> tuple[str, str]:
    """
    Build one (chunk_a, chunk_b) pair at a target semantic-overlap level
    for the given topic dict (one entry of TOPIC_PAIRS).

    chunk_a is always the full base passage. chunk_b keeps a paraphrase
    (never a copy) of the first n_keep sentences of chunk_a and replaces
    the rest with sentences from an unrelated passage, so the two chunks
    share exactly overlap_pct percent of their meaning by construction,
    with no guaranteed literal wording in common anywhere.
    """
    n_total = len(topic["base"])
    n_keep  = round(n_total * overlap_pct / 100)
    chunk_a = " ".join(topic["base"])
    chunk_b = " ".join(topic["paraphrase"][:n_keep] + topic["distractor"][n_keep:])
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


def _check_no_truncation(chunk_a: str, chunk_b: str, topic_name: str, overlap_pct: int) -> None:
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
            f"  [WARNING] {topic_name} @ overlap={overlap_pct}%: pair is "
            f"{true_len} tokens, exceeds MAX_LENGTH={MAX_LENGTH}. The tail "
            f"of chunk_b will be truncated away -- this result cannot be "
            f"trusted."
        )


def _check_length_guard(chunk_a: str, chunk_b: str, topic_name: str, overlap_pct: int) -> None:
    """
    Warn if either chunk exceeds CACD's real LENGTH_GUARD. A chunk longer
    than LENGTH_GUARD is protected from being dropped in the real pipeline
    unless its NIS falls below NIS_FLOOR (0.3); if that never happens on
    this gradient, the length guard -- not NIS or cosine_sim -- would be
    the thing actually deciding CACD's outcome for every pair.
    """
    for name, chunk in [("chunk_a", chunk_a), ("chunk_b", chunk_b)]:
        if len(chunk) > LENGTH_GUARD:
            print(
                f"  [WARNING] {topic_name} @ overlap={overlap_pct}%: {name} "
                f"is {len(chunk)} characters, exceeds LENGTH_GUARD="
                f"{LENGTH_GUARD}. This pair would be length-protected in "
                f"the real pipeline regardless of NIS or cosine_sim."
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

def run_topic_pair(topic: dict) -> list[dict]:
    """Run the full overlap gradient for one topic pair, returning one
    result dict per overlap level."""
    rows = []
    for overlap_pct in OVERLAP_LEVELS:
        chunk_a, chunk_b = build_pair(topic, overlap_pct)
        _check_no_truncation(chunk_a, chunk_b, topic["name"], overlap_pct)
        _check_length_guard(chunk_a, chunk_b, topic["name"], overlap_pct)

        cos_sim = compute_cosine_similarity(chunk_a, chunk_b)
        cacd    = compute_nis(chunk_a, chunk_b)

        rows.append({
            "overlap_pct": overlap_pct,
            "cosine_sim":  round(cos_sim, 4),
            "nis":         cacd["nis"],
            "prob_dup":    cacd["prob_dup"],
        })
        print(
            f"  [{overlap_pct:3d}% overlap] "
            f"cosine_sim={cos_sim:.4f}  NIS={cacd['nis']:.4f}  "
            f"prob_dup={cacd['prob_dup']:.4f}"
        )
    return rows


def print_summary_table(rows: list[dict]) -> None:
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


def correlations(rows: list[dict]) -> dict:
    """Pearson r (linear) and Spearman rho (monotonic rank) between each
    signal and the true overlap percentage. Reporting both matters: a
    signal can rank pairs correctly (high rho) while still being highly
    non-linear in its raw values (lower r), or vice versa."""
    overlaps      = np.array([r["overlap_pct"] for r in rows], dtype=float)
    cos_vals      = np.array([r["cosine_sim"] for r in rows])
    one_minus_nis = 1.0 - np.array([r["nis"] for r in rows])

    r_cos, _   = pearsonr(overlaps, cos_vals)
    r_nis, _   = pearsonr(overlaps, one_minus_nis)
    rho_cos, _ = spearmanr(overlaps, cos_vals)
    rho_nis, _ = spearmanr(overlaps, one_minus_nis)

    return {
        "pearson_cosine": r_cos, "pearson_nis": r_nis,
        "spearman_cosine": rho_cos, "spearman_nis": rho_nis,
    }


def main():
    print("=" * 84)
    print("NIS vs. Cosine Similarity across a controlled semantic-overlap gradient")
    print(f"Cross-encoder: {CROSS_ENCODER_MODEL}")
    print(f"Bi-encoder:    {EMBED_MODEL}")
    print(f"Topic pairs:   {len(TOPIC_PAIRS)}")
    print("=" * 84)

    all_corrs = []
    for topic in TOPIC_PAIRS:
        print()
        print("-" * 84)
        print(f"Topic pair: {topic['name']}")
        print("-" * 84)
        rows = run_topic_pair(topic)

        print()
        print_summary_table(rows)

        corr = correlations(rows)
        all_corrs.append(corr)
        print()
        print(f"  Pearson r   -- cosine_sim: {corr['pearson_cosine']:.4f}  "
              f"| (1-NIS): {corr['pearson_nis']:.4f}")
        print(f"  Spearman rho-- cosine_sim: {corr['spearman_cosine']:.4f}  "
              f"| (1-NIS): {corr['spearman_nis']:.4f}")

    # ── Averaged across all topic pairs ─────────────────────────────────────
    print()
    print("=" * 84)
    print(f"AVERAGED ACROSS {len(TOPIC_PAIRS)} TOPIC PAIRS")
    print("=" * 84)
    for key in ["pearson_cosine", "pearson_nis", "spearman_cosine", "spearman_nis"]:
        vals = [c[key] for c in all_corrs]
        mean = np.mean(vals)
        spread = f"(individual: {', '.join(f'{v:.4f}' for v in vals)})"
        print(f"  {key:<16}: mean = {mean:.4f}  {spread}")

    print()
    print("Interpretation:")
    print("  chunk_b is built from paraphrases at every overlap level, never")
    print("  literal copies, so any lexical overlap between chunk_a and")
    print("  chunk_b is incidental, not by design. This is the case a pooled")
    print("  bi-encoder vector is most at risk of misjudging: two chunks can")
    print("  share little surface wording while still meaning the same thing.")
    print("  Pearson r measures whether a signal's raw values scale linearly")
    print("  with true overlap; Spearman rho measures whether a signal ranks")
    print("  pairs in the correct order, regardless of whether that")
    print("  relationship is linear. A signal can be perfectly ranked (rho =")
    print("  1) while still being highly compressed or non-linear in its raw")
    print("  values (lower r) -- check both before concluding either signal")
    print("  is 'better' from a single number. With only two topic pairs")
    print("  here, treat the averaged numbers as a first look, not a settled")
    print("  result; more topic pairs would make this more conclusive.")
    print("  'Sim >= 0.8?' marks where the Similarity baseline's fixed")
    print("  threshold would call the pair a duplicate at each overlap")
    print("  level; a threshold that stays DUP across most of the 50-100%")
    print("  range would fail to separate near-duplicate chunks from only")
    print("  loosely related ones.")


if __name__ == "__main__":
    main()