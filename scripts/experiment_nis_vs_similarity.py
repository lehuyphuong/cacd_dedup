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

# Overlap levels to test: 100%, 95%, 90%, ..., 50% (11 levels)
OVERLAP_LEVELS = list(range(100, 45, -5))

# Similarity baseline threshold used elsewhere in this paper's experiments,
# shown here only to mark where cosine similarity would call two chunks
# duplicates at each overlap level.
SIMILARITY_THRESHOLD = 0.8

# ── Controlled-overlap chunk pairs (semantic, not verbatim) ────────────────

# Chunk A is always the concatenation of all 20 sentences below.
BASE_FACTS = [
    "The Eiffel Tower, an iconic iron lattice structure, stands prominently on the Champ de Mars in central Paris, France.",
    "French engineer Gustave Eiffel and his company were responsible for designing and constructing this famous landmark.",
    "Work on the tower's foundations and ironwork officially commenced in January of 1887, following years of planning.",
    "After more than two years of construction, the tower was finally completed and opened in March 1889.",
    "The structure was originally built to serve as the entrance arch for the 1889 World's Fair, held in Paris.",
    "Rising to a height of approximately 330 meters, the tower was an extraordinary engineering achievement for its era.",
    "For over four decades, until 1930, it held the record as the tallest man-made structure anywhere in the world.",
    "The entire framework of the tower is constructed from puddled wrought iron, chosen for its strength and relative lightness.",
    "In total, the finished structure weighs approximately ten thousand one hundred metric tons, excluding non-structural elements.",
    "Visitors can access the tower via three distinct public levels, each offering different views of the surrounding city.",
    "Each year, the tower draws roughly seven million visitors from around the world, making it a major tourist destination.",
    "It is widely regarded as one of the most instantly recognizable landmarks anywhere on the planet.",
    "To protect it from corrosion, the tower undergoes a fresh coat of paint approximately once every seven years.",
    "A single repainting effort requires roughly sixty tons of specially formulated paint applied by hand.",
    "During periods of strong wind, the upper sections of the tower can sway slightly from side to side.",
    "Interestingly, the structure was originally intended to be a temporary installation, dismantled after twenty years.",
    "The tower's iron framework comprises what is often described as 108 stories when counted architecturally.",
    "At its base, the structure forms a square measuring approximately 125 meters along each side.",
    "After dark, the tower is illuminated by roughly twenty thousand individual light bulbs, creating a sparkling effect.",
    "Today, the tower endures as an enduring global symbol representing France and its cultural heritage.",
]

# Genuine paraphrases of BASE_FACTS, same order, same meaning, deliberately
# different wording and sentence structure -- this is the "semantic overlap"
# content used to build chunk B, never a copy of BASE_FACTS.
PARAPHRASED_FACTS = [
    "Standing tall in the heart of Paris on the Champ de Mars, the Eiffel Tower is a well-known lattice-work iron landmark.",
    "The landmark's design and construction were carried out by the French engineering firm led by Gustave Eiffel.",
    "Building work on the tower's base and metal frame began at the start of 1887, after extensive preparation.",
    "It took over two years to build, and the tower opened to the public in March of 1889.",
    "Originally, the tower served as a grand gateway for visitors attending the World's Fair hosted in Paris in 1889.",
    "Reaching roughly 330 meters into the sky, the tower represented a remarkable feat of engineering at the time.",
    "The tower remained the world's tallest man-made structure for more than 40 years, losing that title only in 1930.",
    "Wrought iron, valued for being both sturdy and comparatively light, was used throughout the tower's entire framework.",
    "Not counting smaller add-ons, the completed tower has a total weight of around 10,100 metric tons.",
    "The tower offers three separate levels open to the public, each providing a unique vantage point over Paris.",
    "About seven million tourists travel to see the tower annually, making it one of the world's top attractions.",
    "Few structures anywhere are as instantly identifiable as this landmark, which is famous the world over.",
    "Roughly every seven years, workers repaint the entire tower to keep it from rusting.",
    "Each repainting job uses close to sixty tons of paint, applied entirely by hand.",
    "When winds are strong, the tower's upper portion has been known to sway noticeably from one side to the other.",
    "Originally, engineers planned for the tower to stand for only two decades before being taken down.",
    "Architecturally speaking, the tower's iron structure is often said to contain 108 individual stories.",
    "The tower's foundation forms a square shape, with each side stretching about 125 meters.",
    "At night, around twenty thousand light bulbs illuminate the tower, giving it a shimmering appearance.",
    "The tower continues to stand today as a lasting emblem of French culture and identity.",
]

# Unrelated sentences used to replace PARAPHRASED_FACTS sentences in chunk B
# as the target overlap decreases, keeping chunk length roughly constant so
# overlap percentage is not confounded with chunk length.
DISTRACTOR_FACTS = [
    "The Amazon rainforest stretches across a vast portion of northwestern Brazil, forming one of the planet's largest ecosystems.",
    "Beyond Brazil, the forest also extends into neighboring countries including Peru, Colombia, and several other South American nations.",
    "In total, the rainforest covers an area of roughly 5.5 million square kilometers of dense tropical vegetation.",
    "It is widely recognized as the largest tropical rainforest anywhere on Earth, unmatched in scale.",
    "The mighty Amazon River winds its way through the heart of the forest, feeding countless tributaries along the way.",
    "By volume, the Amazon River discharges more freshwater into the ocean than any other river system on the planet.",
    "Millions of distinct plant, animal, and insect species make their home within the boundaries of this rainforest.",
    "Scientists estimate the forest contains roughly ten percent of all species currently known to science.",
    "Numerous indigenous communities have lived within the rainforest for generations, relying on it for their way of life.",
    "The forest plays an outsized role in regulating weather patterns and climate conditions across the globe.",
    "Through photosynthesis, the rainforest is responsible for producing close to twenty percent of the world's oxygen supply.",
    "Each year, significant portions of the rainforest are lost to deforestation driven by human activity.",
    "The primary drivers behind this forest loss are commercial logging operations and large-scale agricultural expansion.",
    "In some areas, the forest canopy rises to heights exceeding forty meters above the ground.",
    "Certain regions of the rainforest receive more than two thousand millimeters of rainfall over the course of a year.",
    "The forest is also an important habitat for thousands of distinct bird species found nowhere else.",
    "Iconic animals such as jaguars, sloths, and river dolphins all make their home within this ecosystem.",
    "Researchers continue to identify previously unknown species living within the depths of the rainforest.",
    "Numerous conservation initiatives have been launched in an effort to preserve what remains of the forest.",
    "Because of its role in producing oxygen, the Amazon is frequently referred to as the lungs of the planet.",
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
print("Models loaded.\n")


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
        return_tensors="pt", truncation=True, max_length=256, padding=True,
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