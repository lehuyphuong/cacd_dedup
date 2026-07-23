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
given topic pair. Chunk B keeps a paraphrase (never a copy) of overlap_pct
percent of chunk A's sentences and replaces the rest with sentences from
an unrelated passage. WHICH sentences are replaced follows a fixed,
reproducible scatter order rather than always cutting off the tail, so
the replaced sentences are spread throughout chunk B instead of
concentrated in one contiguous block -- this was changed after an earlier
block-concatenation design showed prob_dup staying flat across most of
the gradient and then dropping abruptly near 0% overlap, rather than
declining gradually; scattering removes the single sharp "topic breaks
here" point a block design creates, so it isolates whether that
abruptness came from the block structure or is a property of the
classifier itself. See _removal_order and build_pair for details.

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

# Second cross-encoder, trained on STS-Benchmark (continuous 0-5 similarity
# labels) rather than MS MARCO passage-relevance ranking (mostly binary
# relevant/not-relevant). Added to test whether the "confident, then a
# sudden cliff" behavior seen from CACD's cross-encoder is a property of
# cross-encoders in general, or specific to a model trained for ranking
# rather than graded similarity.
CROSS_ENCODER_MODEL_ALT = "cross-encoder/stsb-roberta-base"

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
tokenizer_alt, cross_encoder_alt = _load_cross_encoder(CROSS_ENCODER_MODEL_ALT)

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

def compute_nis(tok, model, chunk_a: str, chunk_b: str) -> dict:
    """
    Score (chunk_a, chunk_b) with the given cross-encoder and compute the
    New Information Score exactly as defined in CACD: entropy of the B=>A
    attention (how much of chunk_b's tokens are explained by chunk_a),
    normalized by log|A|, averaged over the tokens of chunk_b.

    Works with any BERT-family sequence-classification cross-encoder that
    returns attentions, not just CACD_CROSS_ENCODER_MODEL -- used here to
    compare CACD's actual cross-encoder against an STS-trained alternative.

    Returns dict with prob_dup (cross-encoder duplicate probability) and
    nis (New Information Score, in [0, 1]).
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

    last_attn = outputs.attentions[-1][0]      # (num_heads, seq, seq)
    avg_attn  = last_attn.mean(dim=0)          # (seq, seq)

    input_ids = inputs["input_ids"][0]
    n_tokens  = int(inputs["attention_mask"][0].sum().item())
    sep_id    = tok.sep_token_id
    sep_pos   = (input_ids == sep_id).nonzero(as_tuple=True)[0]

    if len(sep_pos) > 0:
        first_sep = int(sep_pos[0].item())
        # Some tokenizers (e.g. RoBERTa: "<s> A </s></s> B </s>") place two
        # consecutive separator tokens between segments, not one (BERT:
        # "[CLS] A [SEP] B [SEP]"). Skip over any run of consecutive sep
        # tokens so b_start lands on the first real token of B in either
        # convention, instead of silently treating a second separator as
        # part of B.
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
    """Run the full overlap gradient for one topic pair under both
    cross-encoders, returning one result dict per overlap level."""
    rows = []
    for overlap_pct in OVERLAP_LEVELS:
        chunk_a, chunk_b = build_pair(topic, overlap_pct)
        _check_no_truncation(tokenizer, chunk_a, chunk_b, topic["name"], overlap_pct, "msmarco")
        _check_no_truncation(tokenizer_alt, chunk_a, chunk_b, topic["name"], overlap_pct, "stsb")
        _check_length_guard(chunk_a, chunk_b, topic["name"], overlap_pct)

        cos_sim  = compute_cosine_similarity(chunk_a, chunk_b)
        cacd     = compute_nis(tokenizer, cross_encoder, chunk_a, chunk_b)
        cacd_alt = compute_nis(tokenizer_alt, cross_encoder_alt, chunk_a, chunk_b)

        rows.append({
            "overlap_pct": overlap_pct,
            "cosine_sim":  round(cos_sim, 4),
            "nis":         cacd["nis"],
            "prob_dup":    cacd["prob_dup"],
            "nis_alt":     cacd_alt["nis"],
            "prob_dup_alt": cacd_alt["prob_dup"],
        })
        print(
            f"  [{overlap_pct:3d}% overlap] "
            f"cosine_sim={cos_sim:.4f}  "
            f"msmarco(NIS={cacd['nis']:.4f}, p_dup={cacd['prob_dup']:.4f})  "
            f"stsb(NIS={cacd_alt['nis']:.4f}, p_dup={cacd_alt['prob_dup']:.4f})"
        )
    return rows


def run_hard_negative_test() -> list[dict]:
    """
    Score each hard-negative pair with cosine similarity and with CACD's
    actual cross-encoder (NIS and prob_dup). Only the msmarco model is used
    here, since this result is meant to describe CACD as it actually is,
    not an alternative model.
    """
    rows = []
    for pair in HARD_NEGATIVE_PAIRS:
        chunk_a, chunk_b = build_hard_negative_pair(pair)
        _check_no_truncation(tokenizer, chunk_a, chunk_b, pair["name"], -1, "msmarco")
        _check_length_guard(chunk_a, chunk_b, pair["name"], -1)

        cos_sim = compute_cosine_similarity(chunk_a, chunk_b)
        cacd    = compute_nis(tokenizer, cross_encoder, chunk_a, chunk_b)

        rows.append({
            "name":       pair["name"],
            "cosine_sim": round(cos_sim, 4),
            "nis":        cacd["nis"],
            "prob_dup":   cacd["prob_dup"],
        })
        print(f"  {pair['name']}")
        print(
            f"    cosine_sim={cos_sim:.4f} "
            f"({'DUP' if cos_sim >= SIMILARITY_THRESHOLD else 'keep'})  "
            f"NIS={cacd['nis']:.4f}  prob_dup={cacd['prob_dup']:.4f} "
            f"({'DUP' if cacd['prob_dup'] >= SIMILARITY_THRESHOLD else 'keep'})"
        )
    return rows


def print_summary_table(rows: list[dict]) -> None:
    header = (
        f"{'Overlap %':>10} | {'Cosine':>7} | {'Sim>=.8?':>9} | "
        f"{'NIS(mm)':>8} | {'pdup(mm)':>9} | {'NIS(sts)':>9} | {'pdup(sts)':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        sim_flag = "DUP" if r["cosine_sim"] >= SIMILARITY_THRESHOLD else "keep"
        print(
            f"{r['overlap_pct']:>9}% | {r['cosine_sim']:>7.4f} | "
            f"{sim_flag:>9} | {r['nis']:>8.4f} | {r['prob_dup']:>9.4f} | "
            f"{r['nis_alt']:>9.4f} | {r['prob_dup_alt']:>10.4f}"
        )


def correlations(rows: list[dict]) -> dict:
    """Pearson r (linear) and Spearman rho (monotonic rank) between each
    signal and the true overlap percentage. Reporting both matters: a
    signal can rank pairs correctly (high rho) while still being highly
    non-linear in its raw values (lower r), or vice versa.

    Includes prob_dup for both cross-encoders (not just NIS), since the
    central question at this point is whether prob_dup's "confident, then
    a sudden cliff" behavior is specific to CACD's MS MARCO-trained
    cross-encoder or general to cross-encoders repurposed this way."""
    overlaps          = np.array([r["overlap_pct"] for r in rows], dtype=float)
    cos_vals          = np.array([r["cosine_sim"] for r in rows])
    one_minus_nis     = 1.0 - np.array([r["nis"] for r in rows])
    one_minus_nis_alt = 1.0 - np.array([r["nis_alt"] for r in rows])
    prob_dup          = np.array([r["prob_dup"] for r in rows])
    prob_dup_alt      = np.array([r["prob_dup_alt"] for r in rows])

    def _both(vals):
        r_, _   = pearsonr(overlaps, vals)
        rho_, _ = spearmanr(overlaps, vals)
        return r_, rho_

    r_cos, rho_cos     = _both(cos_vals)
    r_nis, rho_nis     = _both(one_minus_nis)
    r_nis_a, rho_nis_a = _both(one_minus_nis_alt)
    r_pd, rho_pd       = _both(prob_dup)
    r_pd_a, rho_pd_a   = _both(prob_dup_alt)

    return {
        "pearson_cosine": r_cos, "spearman_cosine": rho_cos,
        "pearson_nis": r_nis, "spearman_nis": rho_nis,
        "pearson_nis_alt": r_nis_a, "spearman_nis_alt": rho_nis_a,
        "pearson_probdup": r_pd, "spearman_probdup": rho_pd,
        "pearson_probdup_alt": r_pd_a, "spearman_probdup_alt": rho_pd_a,
    }


def main():
    print("=" * 92)
    print("NIS vs. Cosine Similarity across a controlled semantic-overlap gradient")
    print(f"Cross-encoder (CACD):     {CROSS_ENCODER_MODEL}")
    print(f"Cross-encoder (STS, alt): {CROSS_ENCODER_MODEL_ALT}")
    print(f"Bi-encoder:               {EMBED_MODEL}")
    print(f"Topic pairs:              {len(TOPIC_PAIRS)}")
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
              f"(1-NIS) mm: {corr['pearson_nis']:.4f} | (1-NIS) sts: {corr['pearson_nis_alt']:.4f} | "
              f"p_dup mm: {corr['pearson_probdup']:.4f} | p_dup sts: {corr['pearson_probdup_alt']:.4f}")
        print(f"  Spearman rho -- cosine: {corr['spearman_cosine']:.4f} | "
              f"(1-NIS) mm: {corr['spearman_nis']:.4f} | (1-NIS) sts: {corr['spearman_nis_alt']:.4f} | "
              f"p_dup mm: {corr['spearman_probdup']:.4f} | p_dup sts: {corr['spearman_probdup_alt']:.4f}")

    # ── Averaged across all topic pairs ─────────────────────────────────────
    print()
    print("=" * 92)
    print(f"AVERAGED ACROSS {len(TOPIC_PAIRS)} TOPIC PAIRS")
    print("=" * 92)
    keys = [
        "pearson_cosine", "pearson_nis", "pearson_nis_alt", "pearson_probdup", "pearson_probdup_alt",
        "spearman_cosine", "spearman_nis", "spearman_nis_alt", "spearman_probdup", "spearman_probdup_alt",
    ]
    for key in keys:
        vals = [c[key] for c in all_corrs]
        mean = np.mean(vals)
        spread = f"(individual: {', '.join(f'{v:.4f}' for v in vals)})"
        print(f"  {key:<22}: mean = {mean:.4f}  {spread}")

    print()
    print("Interpretation:")
    print("  'mm' = CACD's actual cross-encoder (trained on MS MARCO passage")
    print("  ranking, mostly binary relevant/not-relevant); 'sts' = an")
    print("  alternative cross-encoder trained on STS-Benchmark (continuous")
    print("  0-5 similarity labels). If p_dup(sts) correlates with true")
    print("  overlap noticeably better than p_dup(mm), that points to the")
    print("  training objective, not the cross-encoder architecture or CACD's")
    print("  design, as the source of the earlier flat-then-cliff behavior --")
    print("  a model trained for graded similarity should be better")
    print("  calibrated across partial overlap than one trained mostly for")
    print("  binary relevance ranking. If both still show the same cliff")
    print("  pattern, that points more toward the paraphrase/distractor")
    print("  sentence style itself, or the fact-list construction, rather")
    print("  than the specific cross-encoder used.")
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
            f"NIS={r100['nis']:.4f}  prob_dup={r100['prob_dup']:.4f}"
        )

    print()
    print("Interpretation:")
    print("  If cosine_sim scores the hard-negative pairs nearly as high as")
    print("  the genuine-duplicate reference point, that is the false-positive")
    print("  failure mode this paper motivates CACD with: two chunks that")
    print("  share surface style and structure, not real content, being")
    print("  judged as near-duplicates. Whether NIS and prob_dup avoid that")
    print("  same mistake here is the actual test of whether CACD's")
    print("  cross-encoder step adds value beyond pooled similarity, not the")
    print("  overlap-gradient result above. If NIS/prob_dup are fooled here")
    print("  too, that is a genuine limitation to report, not a reason to")
    print("  change the pairs.")


if __name__ == "__main__":
    main()