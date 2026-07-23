"""
scripts/experiment_probdup_vs_similarity.py

Standalone experiment: compares prob_dup (CACD's cross-encoder duplicate
probability, from its classification head) against cosine similarity (the
Similarity baseline's signal, from a bi-encoder's pooled vectors), as two
complete approaches to detecting duplicate chunks.

Why prob_dup vs. cosine_sim, and not NIS vs. cosine_sim: prob_dup and
cosine_sim are directly comparable -- both are trained, general-purpose
"how much do these two chunks overlap" signals, one from a cross-encoder's
classification head, one from a bi-encoder's pooled vectors. NIS is a
different kind of signal. In CACD's actual decision rule
(src/dedup/stage3_decision.py), NIS is only consulted when prob_dup falls
in the uncertain zone (PROB_LOW, PROB_HIGH); it answers a narrower
question -- "does this specific candidate still leave something
unexplained, so the new chunk should be kept" -- not "how similar are
these two chunks overall". Putting NIS head-to-head against cosine_sim as
if they were two general redundancy scores does not reflect how CACD
actually uses it, so this script does not compute NIS at all.

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
than a general property of either signal. TOPIC_PAIRS below holds several
independent (base, paraphrase, distractor) triples; results are reported
per pair and averaged across pairs.

How overlap is controlled: chunk A is always the same base passage for a
given topic pair. Chunk B keeps a paraphrase (never a copy) of overlap_pct
percent of chunk A's sentences and replaces the rest with sentences from
an unrelated passage. WHICH sentences are replaced follows a fixed,
reproducible scatter order rather than always cutting off the tail, so
the replaced sentences are spread throughout chunk B instead of
concentrated in one contiguous block. See _removal_order and build_pair.

HARD_NEGATIVE_PAIRS tests a different, arguably more important failure
mode: two complete, independent passages about two DIFFERENT real things,
written with matching sentence templates, so they share surface style and
structure but are not duplicates at all. A good redundancy signal should
score these low.

Sentence and chunk length: kept short (~7-8 words per sentence, whole
chunk under ~300 characters) for two independent reasons found while
developing this script:
  1. Longer-sentence versions silently exceeded the tokenizer's
     MAX_LENGTH, truncating away the differentiating content near the
     end of chunk_b for most overlap levels. This script asserts at
     runtime that no truncation occurs (see _check_no_truncation).
  2. Chunks longer than CACD's real LENGTH_GUARD (300 characters) are
     protected from being dropped whenever NIS > NIS_FLOOR (0.3) in the
     actual pipeline, which would make the length guard -- not prob_dup
     or cosine_sim -- the thing deciding the outcome. This script also
     asserts at runtime that no chunk exceeds LENGTH_GUARD (see
     _check_length_guard).
Do not trust any result printed alongside either warning.

No dependency on the benchmark pipeline -- only requires:
  pip install transformers sentence-transformers torch numpy scipy

Run from the project root:
  python scripts/experiment_probdup_vs_similarity.py
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

# Max sequence length for the cross-encoder pair and the bi-encoder. 512 is
# the hard architectural ceiling for both models here (standard BERT-family
# position embeddings) -- it cannot be raised further; the fix for
# exceeding it is to shorten the input text, not raise this value.
MAX_LENGTH = 512

# Overlap levels to test per topic pair: 100%, 90%, ..., 0% (11 levels,
# 10% steps).
OVERLAP_LEVELS = list(range(100, -5, -10))

# CACD's real length-aware guard (Section III-D / stage3_decision.py): a
# chunk longer than this is protected from being dropped unless its NIS
# falls below NIS_FLOOR. All chunks built below are kept under this
# threshold (checked at runtime) so that guard cannot mask the comparison.
LENGTH_GUARD = 300

# Similarity baseline threshold used elsewhere in this paper's experiments.
SIMILARITY_THRESHOLD = 0.8

# CACD's own PROB_HIGH threshold (Section III-D): prob_dup >= this value
# puts a candidate in Zone 1, the "confident duplicate" zone, in CACD's
# actual decision rule. Shown alongside SIMILARITY_THRESHOLD so both
# signals are read against the exact cutoff each one actually uses in
# production, not an arbitrary shared number.
PROB_HIGH = 0.8

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
            "Workers collect nectar and pollen.",
            "Bees dance to communicate.",
            "Hives keep steady heat.",
            "Bees make honey for energy.",
            "Colonies can last for years.",
            "Bees pollinate many crops.",
            "Pesticides harm bee colonies.",
            "Beekeepers tend hives with care.",
        ],
    },
]

for _tp in TOPIC_PAIRS:
    assert len(_tp["base"]) == len(_tp["paraphrase"]) == len(_tp["distractor"]), \
        f"Topic pair '{_tp['name']}': base/paraphrase/distractor must be the same length."


# ── Hard-negative pairs ──────────────────────────────────────────────────────
#
# TOPIC_PAIRS above tests a gradient of true overlap (same entity,
# decreasing shared content). It does not test the specific failure mode
# this paper motivates CACD with: two chunks that share surface style,
# structure, and the kind of numbers used, but describe two different
# entities and are not duplicates at all. Each pair below is two
# independent, fully-formed passages about a different real thing, written
# with matching sentence templates (same order of facts: location,
# builder, start date, ...). A good redundancy signal should score these
# LOW.

HARD_NEGATIVE_PAIRS = [
    {
        "name": "Eiffel Tower vs. Notre-Dame Cathedral (same city, different landmark)",
        "chunk_a": [
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
        "chunk_b": [
            "The cathedral is in Paris.",
            "It was built by medieval masons.",
            "Work began in 1163.",
            "It was finished in 1345.",
            "It stands 96 meters tall.",
            "It was Gothic in style.",
            "It is made of stone.",
            "It suffered a fire in 2019.",
            "Millions visit each year.",
            "It symbolizes French heritage.",
        ],
    },
    {
        "name": "Great Wall of China vs. Hadrian's Wall (same concept, different wall)",
        "chunk_a": [
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
        "chunk_b": [
            "Hadrian's Wall is in Britain.",
            "It was built to mark a border.",
            "Building began in 122 AD.",
            "Romans manned the wall.",
            "It spans about 120 kilometers.",
            "Sections use stone and turf.",
            "Forts line the wall.",
            "Thousands of soldiers built it.",
            "Tourists visit it yearly.",
            "It symbolizes Roman Britain.",
        ],
    },
]


def build_hard_negative_pair(pair: dict) -> tuple[str, str]:
    """Join a hard-negative pair's two independent passages into
    (chunk_a, chunk_b), with no overlap-percentage construction involved --
    both are fixed, complete passages about two different real things."""
    return " ".join(pair["chunk_a"]), " ".join(pair["chunk_b"])


def _removal_order(n_total: int, seed: int = 42) -> list[int]:
    """
    A fixed, reproducible random permutation of sentence positions, used to
    decide WHICH positions get replaced by distractor content first as
    overlap decreases. Using a shuffled order instead of always removing
    from the end scatters the distractor sentences throughout chunk_b
    rather than concentrating them in one block at the tail, so there is
    no single sharp "topic breaks here" point partway through the passage.
    """
    import random
    order = list(range(n_total))
    random.Random(seed).shuffle(order)
    return order


def build_pair(topic: dict, overlap_pct: int) -> tuple[str, str]:
    """
    Build one (chunk_a, chunk_b) pair at a target semantic-overlap level
    for the given topic dict (one entry of TOPIC_PAIRS).

    chunk_a is always the full base passage. chunk_b keeps a paraphrase
    (never a copy) of overlap_pct percent of chunk_a's sentences and
    replaces the rest with sentences from an unrelated passage, so the two
    chunks share exactly that percentage of their meaning by construction,
    with no guaranteed literal wording in common anywhere. WHICH sentences
    are replaced follows the fixed scatter order from _removal_order.
    Higher overlap levels are a strict superset of lower ones.
    """
    n_total  = len(topic["base"])
    n_keep   = round(n_total * overlap_pct / 100)
    n_remove = n_total - n_keep
    removed_positions = set(_removal_order(n_total)[:n_remove])

    chunk_a = " ".join(topic["base"])
    chunk_b_sentences = [
        topic["distractor"][i] if i in removed_positions else topic["paraphrase"][i]
        for i in range(n_total)
    ]
    chunk_b = " ".join(chunk_b_sentences)
    return chunk_a, chunk_b


# ── Load models ──────────────────────────────────────────────────────────────

print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL} on {DEVICE} ...")
tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
cross_encoder = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL)
cross_encoder.to(DEVICE)
cross_encoder.eval()

print(f"Loading bi-encoder: {EMBED_MODEL} on {DEVICE} ...")
bi_encoder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
bi_encoder.max_seq_length = MAX_LENGTH
print("Models loaded.\n")


def _check_no_truncation(chunk_a: str, chunk_b: str, name: str, overlap_pct: int) -> None:
    """
    Warn loudly if MAX_LENGTH is too small for this pair, instead of
    letting the tokenizer silently drop the tail of chunk_b -- exactly the
    bug that made an earlier version of this script report an identical
    prob_dup and cosine_sim across most overlap levels (the
    differentiating content near the end of chunk_b never reached the
    model).
    """
    true_len = len(tokenizer(chunk_a, chunk_b, truncation=False)["input_ids"])
    if true_len > MAX_LENGTH:
        print(
            f"  [WARNING] {name} @ overlap={overlap_pct}%: pair is "
            f"{true_len} tokens, exceeds MAX_LENGTH={MAX_LENGTH}. The tail "
            f"of chunk_b will be truncated away -- this result cannot be "
            f"trusted."
        )


def _check_length_guard(chunk_a: str, chunk_b: str, name: str, overlap_pct: int) -> None:
    """
    Warn if either chunk exceeds CACD's real LENGTH_GUARD. A chunk longer
    than LENGTH_GUARD is protected from being dropped in the real pipeline
    unless its NIS falls below NIS_FLOOR (0.3); if that never happens, the
    length guard -- not prob_dup or cosine_sim -- would be the thing
    actually deciding CACD's outcome for that pair.
    """
    for label, chunk in [("chunk_a", chunk_a), ("chunk_b", chunk_b)]:
        if len(chunk) > LENGTH_GUARD:
            print(
                f"  [WARNING] {name} @ overlap={overlap_pct}%: {label} is "
                f"{len(chunk)} characters, exceeds LENGTH_GUARD="
                f"{LENGTH_GUARD}. This pair would be length-protected in "
                f"the real pipeline regardless of prob_dup or cosine_sim."
            )


# ── Scoring functions ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_prob_dup(chunk_a: str, chunk_b: str) -> float:
    """
    Score (chunk_a, chunk_b) with CACD's cross-encoder and return only
    prob_dup, its duplicate probability, exactly as computed in
    src/dedup/stage2_cross_attention.py's score_pair.
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
    return round(prob_dup, 4)


def compute_cosine_similarity(chunk_a: str, chunk_b: str) -> float:
    """Cosine similarity between chunk_a and chunk_b under the same
    bi-encoder used by the Similarity baseline (L2-normalized dense
    embeddings, dot product = cosine similarity)."""
    vecs = bi_encoder.encode([chunk_a, chunk_b], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


# ── Gradient test ────────────────────────────────────────────────────────────

def run_topic_pair(topic: dict) -> list[dict]:
    """Run the full overlap gradient for one topic pair, returning one
    result dict per overlap level."""
    rows = []
    for overlap_pct in OVERLAP_LEVELS:
        chunk_a, chunk_b = build_pair(topic, overlap_pct)
        _check_no_truncation(chunk_a, chunk_b, topic["name"], overlap_pct)
        _check_length_guard(chunk_a, chunk_b, topic["name"], overlap_pct)

        cos_sim  = compute_cosine_similarity(chunk_a, chunk_b)
        prob_dup = compute_prob_dup(chunk_a, chunk_b)

        rows.append({
            "overlap_pct": overlap_pct,
            "cosine_sim":  round(cos_sim, 4),
            "prob_dup":    prob_dup,
        })
        print(f"  [{overlap_pct:3d}% overlap] cosine_sim={cos_sim:.4f}  prob_dup={prob_dup:.4f}")
    return rows


def print_summary_table(rows: list[dict]) -> None:
    header = f"{'Overlap %':>10} | {'Cosine':>7} | {'Sim>=.8?':>9} | {'prob_dup':>9} | {'CACD Zone1?':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        sim_flag = "DUP" if r["cosine_sim"] >= SIMILARITY_THRESHOLD else "keep"
        pd_flag  = "DUP" if r["prob_dup"] >= PROB_HIGH else "keep"
        print(
            f"{r['overlap_pct']:>9}% | {r['cosine_sim']:>7.4f} | {sim_flag:>9} | "
            f"{r['prob_dup']:>9.4f} | {pd_flag:>12}"
        )


def correlations(rows: list[dict]) -> dict:
    """Pearson r (linear) and Spearman rho (monotonic rank) between each
    signal and the true overlap percentage. Reporting both matters: a
    signal can rank pairs correctly (high rho) while still being highly
    non-linear in its raw values (lower r), or vice versa."""
    overlaps = np.array([r["overlap_pct"] for r in rows], dtype=float)
    cos_vals = np.array([r["cosine_sim"] for r in rows])
    pd_vals  = np.array([r["prob_dup"] for r in rows])

    r_cos, _ = pearsonr(overlaps, cos_vals)
    r_pd, _  = pearsonr(overlaps, pd_vals)
    rho_cos, _ = spearmanr(overlaps, cos_vals)
    rho_pd, _  = spearmanr(overlaps, pd_vals)

    return {
        "pearson_cosine": r_cos, "spearman_cosine": rho_cos,
        "pearson_probdup": r_pd, "spearman_probdup": rho_pd,
    }


# ── Hard-negative test ───────────────────────────────────────────────────────

def run_hard_negative_test() -> list[dict]:
    """Score each hard-negative pair with cosine similarity and prob_dup."""
    rows = []
    for pair in HARD_NEGATIVE_PAIRS:
        chunk_a, chunk_b = build_hard_negative_pair(pair)
        _check_no_truncation(chunk_a, chunk_b, pair["name"], -1)
        _check_length_guard(chunk_a, chunk_b, pair["name"], -1)

        cos_sim  = compute_cosine_similarity(chunk_a, chunk_b)
        prob_dup = compute_prob_dup(chunk_a, chunk_b)

        rows.append({"name": pair["name"], "cosine_sim": round(cos_sim, 4), "prob_dup": prob_dup})
        print(f"  {pair['name']}")
        print(
            f"    cosine_sim={cos_sim:.4f} ({'DUP' if cos_sim >= SIMILARITY_THRESHOLD else 'keep'})  "
            f"prob_dup={prob_dup:.4f} ({'DUP' if prob_dup >= PROB_HIGH else 'keep'})"
        )
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 88)
    print("prob_dup (cross-encoder) vs. cosine similarity (bi-encoder)")
    print("across a controlled semantic-overlap gradient and a hard-negative test")
    print(f"Cross-encoder: {CROSS_ENCODER_MODEL}")
    print(f"Bi-encoder:    {EMBED_MODEL}")
    print(f"Topic pairs:   {len(TOPIC_PAIRS)}")
    print("=" * 88)

    all_corrs = []
    all_row_cache = []
    for topic in TOPIC_PAIRS:
        print()
        print("-" * 88)
        print(f"Topic pair: {topic['name']}")
        print("-" * 88)
        rows = run_topic_pair(topic)
        all_row_cache.append(rows)

        print()
        print_summary_table(rows)

        corr = correlations(rows)
        all_corrs.append(corr)
        print()
        print(f"  Pearson r    -- cosine: {corr['pearson_cosine']:.4f} | prob_dup: {corr['pearson_probdup']:.4f}")
        print(f"  Spearman rho -- cosine: {corr['spearman_cosine']:.4f} | prob_dup: {corr['spearman_probdup']:.4f}")

    # ── Averaged across all topic pairs ─────────────────────────────────────
    print()
    print("=" * 88)
    print(f"AVERAGED ACROSS {len(TOPIC_PAIRS)} TOPIC PAIRS")
    print("=" * 88)
    for key in ["pearson_cosine", "pearson_probdup", "spearman_cosine", "spearman_probdup"]:
        vals = [c[key] for c in all_corrs]
        mean = np.mean(vals)
        spread = f"(individual: {', '.join(f'{v:.4f}' for v in vals)})"
        print(f"  {key:<18}: mean = {mean:.4f}  {spread}")

    print()
    print("Interpretation:")
    print("  cosine_sim and prob_dup are two complete, general-purpose")
    print("  duplicate-detection signals: pooled bi-encoder similarity vs. a")
    print("  cross-encoder's own classification output. This compares them")
    print("  directly, each against its own real threshold (cosine_sim >=")
    print("  0.8 for the Similarity baseline; prob_dup >= 0.8 for CACD's")
    print("  Zone 1, the confident-duplicate zone). NIS is not included: it")
    print("  is not a general redundancy score in CACD, only a tie-breaker")
    print("  consulted when prob_dup falls between the two thresholds, so")
    print("  comparing it head-to-head against cosine_sim would not reflect")
    print("  how CACD actually uses it. With only two topic pairs, treat")
    print("  all averaged numbers as a first look, not a settled result.")

    # ── Hard-negative test ───────────────────────────────────────────────────
    print()
    print("=" * 88)
    print("HARD-NEGATIVE TEST: same style/structure, genuinely different entities")
    print("=" * 88)
    print("Each pair below is two complete, independent passages about two")
    print("different real things, written with matching sentence templates.")
    print("Neither is a duplicate of the other.")
    print()
    hn_rows = run_hard_negative_test()

    print()
    print("Reference point -- a genuine duplicate pair (100% overlap, from the")
    print("gradient test above) for comparison:")
    for topic, rows in zip(TOPIC_PAIRS, all_row_cache):
        r100 = next(r for r in rows if r["overlap_pct"] == 100)
        print(f"  {topic['name']}: cosine_sim={r100['cosine_sim']:.4f}  prob_dup={r100['prob_dup']:.4f}")

    print()
    print("Interpretation:")
    print("  If cosine_sim or prob_dup scores a hard-negative pair nearly as")
    print("  high as the genuine-duplicate reference point, that signal is")
    print("  being fooled by surface style and structure rather than judging")
    print("  actual content overlap -- exactly the false-positive failure")
    print("  mode this paper motivates CACD with. Whichever signal keeps a")
    print("  clear gap between the hard-negative pairs and the reference")
    print("  point is the one that better detects genuine duplicates without")
    print("  being misled by matching sentence templates.")


if __name__ == "__main__":
    main()