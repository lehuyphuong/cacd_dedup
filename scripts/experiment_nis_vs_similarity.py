"""
scripts/experiment_nis_vs_similarity.py

Standalone experiment: compares signals derived from CACD's actual
cross-encoder (cross-encoder/msmarco-MiniLM-L6-en-de-v1) against plain
cosine similarity (the Similarity baseline's signal), across a controlled
gradient of SEMANTIC overlap and a hard-negative test.

This version compares three cross-encoder-derived signals, all from the
same single forward pass, rather than just the last-layer NIS used in
CACD today:
  - nis        : CACD's current definition. Entropy of the B=>A attention
                 from the LAST transformer layer, normalized by log|A|.
  - nis_mid    : the same entropy computation, but read from a MIDDLE
                 transformer layer instead of the last one. Prior work on
                 BERT-family models suggests middle layers tend to carry
                 more semantic/coreference information, while later
                 layers specialize toward the model's fine-tuning
                 objective (here, MS MARCO relevance ranking) -- this
                 tests whether that specialization is why last-layer NIS
                 struggled to tell genuine paraphrases from same-style,
                 different-entity text in an earlier hard-negative test.
  - redundancy_signal : max-alignment coverage (BERTScore-style), already
                 computed in CACD's production code
                 (src/dedup/stage2_cross_attention.py) but not currently
                 used by the decision rule. min(coverage_a_to_b,
                 coverage_b_to_a) from the last layer.

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
than a general property of any signal here. TOPIC_PAIRS below holds several
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
     actual pipeline, which would make the length guard -- not any
     cross-encoder signal or cosine_sim -- the thing deciding the
     outcome. This script also asserts at runtime that no chunk exceeds
     LENGTH_GUARD (see _check_length_guard).
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

# Max sequence length for the cross-encoder pair and the bi-encoder. 512 is
# the hard architectural ceiling for both models here (standard BERT-family
# position embeddings) -- it cannot be raised further; the fix for
# exceeding it is to shorten the input text, not raise this value.
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
# TOPIC_PAIRS above tests a gradient of true overlap (same entity, decreasing
# shared content). It does not test the specific failure mode this paper
# motivates CACD with: two chunks that share surface style, structure, and
# the kind of numbers used, but describe two different entities and are not
# duplicates at all. Each pair below is two independent, fully-formed
# passages about a different real thing, written with matching sentence
# templates (same order of facts: location, builder, start date, ...). A
# good redundancy signal should score these LOW; if cosine similarity scores
# them high while NIS/p_dup do not, that specifically supports why CACD
# compares more than pooled similarity. If NIS/p_dup are also fooled, that
# is a real limitation worth reporting, not a reason to change the pairs.

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
    rather than concentrating them in one block at the tail.

    This directly tests a specific hypothesis: the earlier block-concatenation
    design (keep sentences 0..n_keep-1, replace n_keep..end) put every
    replaced sentence in one contiguous run at the end of chunk_b, creating
    a single sharp "topic breaks here" point partway through the passage.
    A cross-encoder attending over the whole sequence may respond to that
    structural discontinuity itself, not just to the true overlap
    percentage, which could explain why prob_dup stayed flat and then
    dropped abruptly rather than declining gradually. Scattering the
    replacements removes that single breakpoint; if prob_dup still jumps
    abruptly with scattering, the flat-then-cliff behavior is more likely
    an inherent property of this classifier head, not an artifact of
    block placement.
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
    are replaced is decided by a fixed scatter order (_removal_order), not
    by always cutting off the tail, so the replaced sentences are spread
    throughout chunk_b rather than concentrated in one contiguous block.
    Higher overlap levels are a strict superset of lower ones: the
    sentence removed going from 100% to 90% stays removed at every lower
    level too, so the gradient decays monotonically.
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

def _load_cross_encoder(model_name: str):
    """Load one cross-encoder + tokenizer pair, ready for scoring."""
    print(f"Loading cross-encoder: {model_name} on {DEVICE} ...")
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_attentions=True
    )
    model.to(DEVICE)
    model.eval()
    return tok, model


tokenizer, cross_encoder = _load_cross_encoder(CROSS_ENCODER_MODEL)

print(f"Loading bi-encoder: {EMBED_MODEL} on {DEVICE} ...")
bi_encoder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
bi_encoder.max_seq_length = MAX_LENGTH
print("Models loaded.\n")


def _check_no_truncation(tok, chunk_a: str, chunk_b: str, topic_name: str, overlap_pct: int, model_label: str) -> None:
    """
    Warn loudly if MAX_LENGTH is too small for this pair under the given
    tokenizer, instead of letting it silently drop the tail of chunk_b --
    exactly the bug that made an earlier version of this script report an
    identical NIS and cosine_sim across most overlap levels (the
    differentiating content near the end of chunk_b never reached the
    model).
    """
    true_len = len(tok(chunk_a, chunk_b, truncation=False)["input_ids"])
    if true_len > MAX_LENGTH:
        print(
            f"  [WARNING] [{model_label}] {topic_name} @ overlap={overlap_pct}%: "
            f"pair is {true_len} tokens, exceeds MAX_LENGTH={MAX_LENGTH}. "
            f"The tail of chunk_b will be truncated away -- this result "
            f"cannot be trusted."
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

def _entropy_nis(attn: torch.Tensor, a_range: slice, b_range: slice, len_a: int) -> float:
    """
    New Information Score from one layer's attention matrix (heads already
    averaged): entropy of the B=>A attention, normalized by log|A|,
    averaged over tokens of B. Shared by both the last-layer (CACD's
    current definition) and middle-layer computations below, so the two
    are computed identically except for which layer's attention is used.
    """
    sub_b2a = attn[b_range, a_range]
    if sub_b2a.numel() == 0 or len_a < 2:
        return 1.0
    row_sums = sub_b2a.sum(dim=1, keepdim=True).clamp(min=1e-9)
    p_b2a    = sub_b2a / row_sums
    ent      = -(p_b2a * torch.log(p_b2a + 1e-9)).sum(dim=1)
    max_ent  = float(np.log(len_a))
    if max_ent <= 1e-9:
        return 1.0
    return float((ent / max_ent).mean().clamp(0.0, 1.0).item())


def _max_alignment_coverage(attn: torch.Tensor, a_range: slice, b_range: slice) -> tuple[float, float]:
    """
    Max-alignment coverage (BERTScore-style), exactly as implemented in
    CACD's production code (src/dedup/stage2_cross_attention.py's
    _max_alignment_coverage) but not currently read by the decision rule.
    coverage_a_to_b: for each token of A, its single strongest attention
    weight toward any token of B, averaged over A's tokens (and the
    reverse for coverage_b_to_a).
    """
    sub_a2b = attn[a_range, b_range]
    sub_b2a = attn[b_range, a_range]
    if sub_a2b.numel() == 0 or sub_b2a.numel() == 0:
        return 0.0, 0.0
    cov_a2b = sub_a2b.max(dim=1).values.mean().item()
    cov_b2a = sub_b2a.max(dim=1).values.mean().item()
    return cov_a2b, cov_b2a


def compute_nis(tok, model, chunk_a: str, chunk_b: str) -> dict:
    """
    Score (chunk_a, chunk_b) with the cross-encoder in a single forward
    pass, and compute three signals from the resulting attentions:

      nis               : CACD's current definition -- entropy-based NIS
                           from the LAST transformer layer.
      nis_mid           : the same entropy-based NIS, but from a MIDDLE
                           transformer layer instead.
      redundancy_signal : max-alignment coverage from the LAST layer,
                           min(coverage_a_to_b, coverage_b_to_a) -- already
                           computed in CACD's production code but not
                           currently used by the decision rule.

    Also returns prob_dup, the cross-encoder's own duplicate probability.
    """
    inputs = tok(
        chunk_a, chunk_b,
        return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
    logits  = outputs.logits.squeeze()
    prob_dup = (
        float(torch.sigmoid(logits).item())
        if logits.dim() == 0
        else float(torch.softmax(logits, dim=-1)[-1].item())
    )

    all_layers = outputs.attentions              # tuple of (1, num_heads, seq, seq), one per layer
    n_layers   = len(all_layers)
    last_attn  = all_layers[-1][0].mean(dim=0)    # (seq, seq), heads averaged
    mid_idx    = n_layers // 2
    mid_attn   = all_layers[mid_idx][0].mean(dim=0)

    input_ids = inputs["input_ids"][0]
    n_tokens  = int(inputs["attention_mask"][0].sum().item())
    sep_id    = tok.sep_token_id
    sep_pos   = (input_ids == sep_id).nonzero(as_tuple=True)[0]

    if len(sep_pos) > 0:
        first_sep = int(sep_pos[0].item())
        # Some tokenizers place two consecutive separator tokens between
        # segments (RoBERTa-style "</s></s>") rather than one (BERT-style
        # "[SEP]"); skip any run of consecutive sep tokens so b_start lands
        # on the first real token of B either way. CACD's own tokenizer
        # uses the single-separator convention, but this stays defensive.
        b_start = first_sep + 1
        while b_start < n_tokens and int(input_ids[b_start].item()) == sep_id:
            b_start += 1
        sep_idx = first_sep
    else:
        sep_idx = n_tokens // 2
        b_start = sep_idx + 1

    len_a   = sep_idx - 1
    a_range = slice(1, sep_idx)
    b_range = slice(b_start, n_tokens - 1)

    nis     = _entropy_nis(last_attn, a_range, b_range, len_a)
    nis_mid = _entropy_nis(mid_attn, a_range, b_range, len_a)
    cov_a2b, cov_b2a = _max_alignment_coverage(last_attn, a_range, b_range)
    redundancy_signal = min(cov_a2b, cov_b2a)

    return {
        "prob_dup":          round(prob_dup, 4),
        "nis":               round(nis, 4),
        "nis_mid":           round(nis_mid, 4),
        "mid_layer_index":   mid_idx,
        "n_layers":          n_layers,
        "coverage_a_to_b":   round(cov_a2b, 4),
        "coverage_b_to_a":   round(cov_b2a, 4),
        "redundancy_signal": round(redundancy_signal, 4),
    }


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
        _check_no_truncation(tokenizer, chunk_a, chunk_b, topic["name"], overlap_pct, "msmarco")
        _check_length_guard(chunk_a, chunk_b, topic["name"], overlap_pct)

        cos_sim = compute_cosine_similarity(chunk_a, chunk_b)
        cacd    = compute_nis(tokenizer, cross_encoder, chunk_a, chunk_b)

        rows.append({
            "overlap_pct":       overlap_pct,
            "cosine_sim":        round(cos_sim, 4),
            "nis":               cacd["nis"],
            "nis_mid":           cacd["nis_mid"],
            "prob_dup":          cacd["prob_dup"],
            "redundancy_signal": cacd["redundancy_signal"],
        })
        print(
            f"  [{overlap_pct:3d}% overlap] "
            f"cosine_sim={cos_sim:.4f}  NIS(last)={cacd['nis']:.4f}  "
            f"NIS(mid,L{cacd['mid_layer_index']})={cacd['nis_mid']:.4f}  "
            f"redundancy={cacd['redundancy_signal']:.4f}  "
            f"prob_dup={cacd['prob_dup']:.4f}"
        )
    return rows


def run_hard_negative_test() -> list[dict]:
    """
    Score each hard-negative pair with cosine similarity and with CACD's
    actual cross-encoder (NIS from the last and a middle layer, prob_dup,
    and the unused-in-production redundancy_signal).
    """
    rows = []
    for pair in HARD_NEGATIVE_PAIRS:
        chunk_a, chunk_b = build_hard_negative_pair(pair)
        _check_no_truncation(tokenizer, chunk_a, chunk_b, pair["name"], -1, "msmarco")
        _check_length_guard(chunk_a, chunk_b, pair["name"], -1)

        cos_sim = compute_cosine_similarity(chunk_a, chunk_b)
        cacd    = compute_nis(tokenizer, cross_encoder, chunk_a, chunk_b)

        rows.append({
            "name":              pair["name"],
            "cosine_sim":        round(cos_sim, 4),
            "nis":               cacd["nis"],
            "nis_mid":           cacd["nis_mid"],
            "prob_dup":          cacd["prob_dup"],
            "redundancy_signal": cacd["redundancy_signal"],
        })
        print(f"  {pair['name']}")
        print(
            f"    cosine_sim={cos_sim:.4f} "
            f"({'DUP' if cos_sim >= SIMILARITY_THRESHOLD else 'keep'})  "
            f"NIS(last)={cacd['nis']:.4f}  NIS(mid,L{cacd['mid_layer_index']})={cacd['nis_mid']:.4f}  "
            f"redundancy={cacd['redundancy_signal']:.4f}  "
            f"prob_dup={cacd['prob_dup']:.4f} "
            f"({'DUP' if cacd['prob_dup'] >= SIMILARITY_THRESHOLD else 'keep'})"
        )
    return rows


def print_summary_table(rows: list[dict]) -> None:
    header = (
        f"{'Overlap %':>10} | {'Cosine':>7} | {'Sim>=.8?':>9} | "
        f"{'NIS(last)':>9} | {'NIS(mid)':>9} | {'redund.':>8} | {'p_dup':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sim_flag = "DUP" if r["cosine_sim"] >= SIMILARITY_THRESHOLD else "keep"
        print(
            f"{r['overlap_pct']:>9}% | {r['cosine_sim']:>7.4f} | "
            f"{sim_flag:>9} | {r['nis']:>9.4f} | {r['nis_mid']:>9.4f} | "
            f"{r['redundancy_signal']:>8.4f} | {r['prob_dup']:>7.4f}"
        )


def correlations(rows: list[dict]) -> dict:
    """Pearson r (linear) and Spearman rho (monotonic rank) between each
    signal and the true overlap percentage. Reporting both matters: a
    signal can rank pairs correctly (high rho) while still being highly
    non-linear in its raw values (lower r), or vice versa.

    cosine_sim and redundancy_signal both read as "higher = more alike",
    so they are correlated directly. nis and nis_mid read as "higher =
    more novel", so 1 - x is used to put them on the same direction before
    correlating."""
    overlaps           = np.array([r["overlap_pct"] for r in rows], dtype=float)
    cos_vals           = np.array([r["cosine_sim"] for r in rows])
    one_minus_nis      = 1.0 - np.array([r["nis"] for r in rows])
    one_minus_nis_mid  = 1.0 - np.array([r["nis_mid"] for r in rows])
    redundancy         = np.array([r["redundancy_signal"] for r in rows])

    def _both(vals):
        r_, _   = pearsonr(overlaps, vals)
        rho_, _ = spearmanr(overlaps, vals)
        return r_, rho_

    r_cos, rho_cos     = _both(cos_vals)
    r_nis, rho_nis     = _both(one_minus_nis)
    r_nism, rho_nism   = _both(one_minus_nis_mid)
    r_red, rho_red     = _both(redundancy)

    return {
        "pearson_cosine": r_cos, "spearman_cosine": rho_cos,
        "pearson_nis": r_nis, "spearman_nis": rho_nis,
        "pearson_nis_mid": r_nism, "spearman_nis_mid": rho_nism,
        "pearson_redundancy": r_red, "spearman_redundancy": rho_red,
    }


def main():
    print("=" * 92)
    print("Cross-encoder signals vs. Cosine Similarity: last-layer NIS, a middle-layer")
    print("variant, and max-alignment coverage, across a semantic-overlap gradient")
    print(f"Cross-encoder (CACD): {CROSS_ENCODER_MODEL}")
    print(f"Bi-encoder:           {EMBED_MODEL}")
    print(f"Topic pairs:          {len(TOPIC_PAIRS)}")
    print("=" * 92)

    all_corrs = []
    all_row_cache = []
    for topic in TOPIC_PAIRS:
        print()
        print("-" * 92)
        print(f"Topic pair: {topic['name']}")
        print("-" * 92)
        rows = run_topic_pair(topic)
        all_row_cache.append(rows)

        print()
        print_summary_table(rows)

        corr = correlations(rows)
        all_corrs.append(corr)
        print()
        print(f"  Pearson r    -- cosine: {corr['pearson_cosine']:.4f} | "
              f"(1-NIS) last: {corr['pearson_nis']:.4f} | (1-NIS) mid: {corr['pearson_nis_mid']:.4f} | "
              f"redundancy: {corr['pearson_redundancy']:.4f}")
        print(f"  Spearman rho -- cosine: {corr['spearman_cosine']:.4f} | "
              f"(1-NIS) last: {corr['spearman_nis']:.4f} | (1-NIS) mid: {corr['spearman_nis_mid']:.4f} | "
              f"redundancy: {corr['spearman_redundancy']:.4f}")

    # ── Averaged across all topic pairs ─────────────────────────────────────
    print()
    print("=" * 92)
    print(f"AVERAGED ACROSS {len(TOPIC_PAIRS)} TOPIC PAIRS")
    print("=" * 92)
    keys = [
        "pearson_cosine", "pearson_nis", "pearson_nis_mid", "pearson_redundancy",
        "spearman_cosine", "spearman_nis", "spearman_nis_mid", "spearman_redundancy",
    ]
    for key in keys:
        vals = [c[key] for c in all_corrs]
        mean = np.mean(vals)
        spread = f"(individual: {', '.join(f'{v:.4f}' for v in vals)})"
        print(f"  {key:<20}: mean = {mean:.4f}  {spread}")

    print()
    print("Interpretation:")
    print("  'last' = CACD's current NIS definition (entropy from the final")
    print("  transformer layer's attention). 'mid' = the same entropy")
    print("  computation read from a middle layer instead, testing whether")
    print("  the final layer's specialization toward MS MARCO relevance")
    print("  ranking (rather than semantic alignment) explains NIS's")
    print("  earlier difficulty telling genuine paraphrases from same-style,")
    print("  different-entity text. 'redundancy' = max-alignment coverage,")
    print("  already computed in CACD's production code but not currently")
    print("  used by the decision rule -- included here as a second")
    print("  candidate signal, not a replacement for NIS.")
    print("  With only two topic pairs, treat all averaged numbers as a")
    print("  first look, not a settled result.")
    print("  'Sim >= 0.8?' marks where the Similarity baseline's fixed")
    print("  threshold would call the pair a duplicate at each overlap level.")

    # ── Hard-negative test ───────────────────────────────────────────────────
    print()
    print("=" * 92)
    print("HARD-NEGATIVE TEST: same style/structure, genuinely different entities")
    print("=" * 92)
    print("Each pair below is two complete, independent passages about two")
    print("different real things, written with matching sentence templates.")
    print("Neither is a duplicate of the other. Only CACD's actual")
    print("cross-encoder (msmarco) is used here -- this result is meant to")
    print("describe CACD as it actually is.")
    print()
    hn_rows = run_hard_negative_test()

    print()
    print("Reference point -- a genuine duplicate pair (100% overlap, from the")
    print("gradient test above) for comparison:")
    for topic, rows in zip(TOPIC_PAIRS, all_row_cache):
        r100 = next(r for r in rows if r["overlap_pct"] == 100)
        print(
            f"  {topic['name']}: cosine_sim={r100['cosine_sim']:.4f}  "
            f"NIS(last)={r100['nis']:.4f}  NIS(mid)={r100['nis_mid']:.4f}  "
            f"redundancy={r100['redundancy_signal']:.4f}  prob_dup={r100['prob_dup']:.4f}"
        )

    print()
    print("Interpretation:")
    print("  If cosine_sim scores the hard-negative pairs nearly as high as")
    print("  the genuine-duplicate reference point, that is the false-positive")
    print("  failure mode this paper motivates CACD with: two chunks that")
    print("  share surface style and structure, not real content, being")
    print("  judged as near-duplicates. Whether NIS(last), NIS(mid), and")
    print("  redundancy_signal avoid that same mistake here is the actual")
    print("  test of whether reading attention differently recovers a")
    print("  meaningful signal for this failure mode. If all of them are")
    print("  fooled here too, that is a genuine limitation to report, not a")
    print("  reason to change the pairs.")


if __name__ == "__main__":
    main()