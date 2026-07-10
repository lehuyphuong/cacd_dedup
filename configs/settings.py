"""
Central configuration for cacd-dedup.

A standalone project, separated from rag-bench-v4. Retains the same 9
chunking strategies for evaluation, but replaces all 5 filter methods
(NoFilter/ExactNorm/MinHashLSH/Similarity/NERExact) with a single
pipeline: CACD (Cross-Attention Calibrated Deduplication).

CACD pipeline (4 stages, DROP branch only — Merge reserved for future work):
  Stage 0: Embedding          — all-MiniLM-L6-v2, 384-dim
  Stage 1: Coarse retrieval   — batch query Qdrant (HNSW), top-K candidates
  Stage 2: Cross-attention    — cross-encoder/msmarco-MiniLM-L6-en-de-v1,
                                 extract attention matrix, compute NIS
  Stage 3: Decision           — 3-zone logic using prob_duplicate + NIS
                                 + length-aware guard

Evaluation metrics: same 4 metrics as rag-bench-v4 —
  Precision, Recall, IoU, Index Size (chunk count + storage MB).
"""

from pathlib import Path

import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
RESULTS_DIR  = ROOT_DIR / "results"
QDRANT_PATH  = DATA_DIR / "qdrant_storage"
HEATMAP_DIR  = RESULTS_DIR / "heatmaps"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
QDRANT_PATH.mkdir(exist_ok=True)
HEATMAP_DIR.mkdir(exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME       = "rajpurkar/squad"
DATASET_SPLIT      = "validation"
# MAX_DOCUMENTS:       int | None = 500
# MAX_EVAL_QUESTIONS:  int | None = 200

MAX_DOCUMENTS:       int | None = None
MAX_EVAL_QUESTIONS:  int | None = None

# ── Embedding — all-MiniLM-L6-v2 (Stage 0) ───────────────────────────────────
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DIM   = 384
# Larger batch size improves GPU utilisation.
# GPU 8GB+: 512-1024. CPU: keep at 128.
EMBED_BATCH_SIZE = 512 if torch.cuda.is_available() else 128

# ── CACD — Stage 1 (Coarse retrieval) ────────────────────────────────────────
# Number of new chunks scored together in a single Stage 2 forward-pass
# batch during ingestion, instead of one chunk (K=5 pairs) at a time.
# Trade-off: chunks within the same micro-batch are all checked against the
# SAME index snapshot (as of the start of the batch) — they cannot see each
# other, only chunks already in the index before the batch began. Smaller
# values = less staleness, closer to fully sequential; larger values =
# fewer, bigger forward passes = faster, at the cost of a larger blind spot
# within each batch. Set to 1 to reproduce the old fully-sequential behavior
# exactly.
CACD_INGEST_BATCH_SIZE = 128

# Number of (text_a, text_b) pairs sent to the cross-encoder in a single
# forward pass inside score_pairs_batched, regardless of how many chunks
# CACD_INGEST_BATCH_SIZE groups together. If one micro-batch of chunks
# produces MORE pairs than this, score_pairs_batched splits them into
# several forward passes automatically — so if this is set too low, it
# silently caps how much CACD_INGEST_BATCH_SIZE can help. Raise this as far
# as GPU memory allows (attention tensors are O(batch x heads x seq^2), so
# watch VRAM when increasing) to let more chunks' pairs land in one real
# forward pass. 512 is a starting point for a modern GPU with short
# (<=256-token) inputs; lower it if you hit out-of-memory errors.
CACD_SUB_BATCH_SIZE = 512

CACD_TOP_K_CANDIDATES = 5   # top-K nearest neighbours retrieved from HNSW per chunk

# Micro-batch size for Stage 1 + Stage 2: instead of processing chunks one
# at a time (Stage 1 batch=1, Stage 2 batch=K=5), chunks are processed in
# windows of this size. Stage 1 issues ONE batched Qdrant query for the
# whole window (already supported by batch_coarse_retrieve, previously
# called with a 1-chunk list); Stage 2 flattens every (chunk, candidate)
# pair across the whole window into ONE cross-encoder forward pass via
# score_multi_chunks_batched, instead of window_size separate K=5-pair calls.
#
# Trade-off (accepted, see conversation): all chunks within one window are
# checked against the index as it stood at the START of the window — they
# do not see each other, even if an earlier chunk in the same window would
# have been kept. Only the window boundary introduces this staleness;
# chunks in different windows are still fully sequential relative to each
# other. Larger values = fewer, bigger forward passes (faster, more stale);
# smaller values = closer to the original fully-sequential behaviour.
CACD_MICROBATCH_SIZE = 32

# Mixed-precision (FP16) inference for the cross-encoder. Only takes effect
# when running on CUDA (torch.autocast on CPU gives no speedup and is not
# what this flag is for); on CPU-only runs this is a no-op regardless of
# the value below. Expected to noticeably speed up Stage 2 / merge scoring
# on GPU with negligible effect on logits/NIS (values a few decimal places
# off at most — far from enough to flip any KEEP/DROP decision in practice).
CACD_USE_FP16 = True

# ── CACD — Stage 2 (Cross-attention) ─────────────────────────────────────────
# Paper-selected model: cross-encoder/msmarco-MiniLM-L6-en-de-v1, chosen after
# a 37-model comparison experiment; multilingual EN-DE MiniLM-L6.
#
# SPEED EXPERIMENT (abandoned): tried cross-encoder/ms-marco-MiniLM-L4-v2 as a
# faster stand-in. Reverted for two reasons: (1) it is a different model
# family (English-only MS MARCO training, not cross-lingual EN-DE — the
# en-de-v1 family has no L4/L2 sibling to begin with), so its attention maps
# are not directly comparable to what the NIS/heatmaps were designed around;
# (2) PROB_HIGH/PROB_LOW/NIS_DROP_THRESHOLD were calibrated for L6's score
# distribution and did not transfer — drop rate collapsed to ~0.16% on the
# full run (CACD behaving almost like NoFilter), and recalibrating would not
# have fixed the deeper issue of the attention maps themselves being off.
# Back to L6; CACD_INGEST_BATCH_SIZE (below) remains the active speed lever.
CACD_CROSS_ENCODER_MODEL = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── CACD — Stage 3 (Decision: Bayes-optimal cutoff) ──────────────────────────
# cutoff = cost_FP / (cost_FP + cost_FN)
#   cost_FP: cost of incorrectly dropping a non-duplicate chunk (information loss)
#   cost_FN: cost of incorrectly keeping a duplicate chunk (wasted index space)
#
# Default is symmetric (cutoff = 0.5). Increase CACD_COST_FALSE_POSITIVE to
# make the system more conservative when dropping (prefer recall over precision).
CACD_COST_FALSE_POSITIVE = 1.3
CACD_COST_FALSE_NEGATIVE = 0.7

# ── Vector store — Qdrant (embedded, no Docker) ───────────────────────────────
COLLECTION_PREFIX = "cacd_dedup"

# ── Retrieval (final evaluation, top-k = 5 matching rag-bench-v4) ─────────────
TOP_K = 5

# ── LLM (Ollama — optional, not used in current scope) ───────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "mistral"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 256

SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer the question using \
ONLY the provided context passages. Be concise — answer in 1-3 sentences. \
If the answer is not in the context, respond with exactly: "I don't know."\
"""

# ── Chunking configurations ───────────────────────────────────────────────────
#
# 18 chunking configs × 1 CACD pipeline = 18 total configs
#
# Strategy params:
#   chunk_size : target chunk size in characters (base unit)
#   overlap    : character overlap between chunks (0 for strategies that
#                do not use sliding windows)
#   extra      : strategy-specific overrides (optional)

def _make_configs() -> list[dict]:
    chunker_configs = [
        # ── FixedSize == FixedToken (paper chunk_size=200,400) ────────────────
        {"strategy": "FixedSize",   "chunk_size": 200, "overlap": 0},
        {"strategy": "FixedSize",   "chunk_size": 400, "overlap": 0},

        # ── Recursive == RecursiveToken (paper chunk_size=200,400) ────────────
        {"strategy": "Recursive",   "chunk_size": 200, "overlap": 0},
        {"strategy": "Recursive",   "chunk_size": 400, "overlap": 0},

        # ── Semantic == ClusterSemantic (paper chunk_size=200,400) ────────────
        {"strategy": "Semantic",    "chunk_size": 200, "overlap": 0,
         "extra": {"threshold_percentile": 95.0}},
        {"strategy": "Semantic",    "chunk_size": 400, "overlap": 0,
         "extra": {"threshold_percentile": 95.0}},

        # ── Overlapping (paper chunk_size=400/overlap=200, 800/overlap=400) ───
        {"strategy": "Overlapping", "chunk_size": 400, "overlap": 200},
        {"strategy": "Overlapping", "chunk_size": 800, "overlap": 400},

        # ── AdaptiveEntropy ───────────────────────────────────────────────────
        {"strategy": "AdaptiveEntropy",       "chunk_size": 300, "overlap": 0},
        {"strategy": "AdaptiveEntropy",       "chunk_size": 500, "overlap": 0},

        # ── AdaptiveSentenceLen ───────────────────────────────────────────────
        {"strategy": "AdaptiveSentenceLen",   "chunk_size": 4,   "overlap": 0,
         "extra": {"target_sentences": 4, "min_sentences": 2, "max_sentences": 8}},
        {"strategy": "AdaptiveSentenceLen",   "chunk_size": 6,   "overlap": 0,
         "extra": {"target_sentences": 6, "min_sentences": 3, "max_sentences": 12}},

        # ── HierarchicalParentChild ───────────────────────────────────────────
        {"strategy": "HierarchicalParentChild", "chunk_size": 200, "overlap": 0,
         "extra": {"parent_size": 600}},
        {"strategy": "HierarchicalParentChild", "chunk_size": 400, "overlap": 0,
         "extra": {"parent_size": 800}},

        # ── Contextual ────────────────────────────────────────────────────────
        {"strategy": "Contextual",            "chunk_size": 300, "overlap": 0},
        {"strategy": "Contextual",            "chunk_size": 500, "overlap": 0},

        # ── TopicBased ────────────────────────────────────────────────────────
        {"strategy": "TopicBased",            "chunk_size": 200, "overlap": 0,
         "extra": {"n_topics": 4, "min_chunk_size": 100}},
        {"strategy": "TopicBased",            "chunk_size": 400, "overlap": 0,
         "extra": {"n_topics": 6, "min_chunk_size": 200}},
    ]

    configs = []
    for chunker in chunker_configs:
        cfg = {
            "strategy":   chunker["strategy"],
            "chunk_size": chunker["chunk_size"],
            "overlap":    chunker["overlap"],
        }
        if "extra" in chunker:
            cfg["extra"] = chunker["extra"]
        configs.append(cfg)

    return configs


CHUNKING_CONFIGS = _make_configs()

# ── Sentence-level merge safety guards ────────────────────────────────────────
#
# These three settings close the gap identified after analysing the full-dataset
# Merge run (Recall highest, Precision lowest -> IoU lowest of all methods):
#
#   1. MERGE_SAME_DOC_ONLY: the persistent index intentionally spans the whole
#      2,067-document corpus (dedup is meant to catch redundancy *across*
#      documents, not just within one) — that scope is correct by design and
#      is NOT changed here. What was missing is a same-document constraint
#      specifically on the MERGE target: a novel sentence from document A must
#      not be spliced into a chunk that will keep being served as document B's
#      content. Restricting merge (not retrieval/dedup) to same-doc candidates
#      prevents cross-document token contamination in Te/Tr token-overlap eval.
#
#   2. MERGE_MAX_SIZE_MULTIPLIER: caps how large a merged chunk may grow
#      relative to its strategy's target chunk_size, so a single "hub" chunk
#      cannot silently absorb an unbounded number of merges over the course
#      of one ingest pass.
#
#   3. MERGE_MAX_EMBED_CHARS: a hard character ceiling approximating the
#      embedding model's max_seq_length (all-MiniLM-L6-v2 ~= 256 tokens；
#      ~4 chars/token in English, minus margin) so that whatever text is
#      passed to embed_fn() after a merge is fully represented in the new
#      vector rather than silently truncated by the tokenizer. Re-embedding
#      after merge (already done in _sentence_level_merge) only fixes staleness
#      if the text handed to it is short enough to be embedded in full — this
#      cap is what actually guarantees that.
# Master switch for the sentence-level merge step (Section III-G / Algorithm 2).
# Default OFF: merge is the dominant cost driver (many extra per-sentence
# cross-encoder passes on every dropped chunk) and also grows the index
# (appended text), which is why CACD-with-merge ended up larger and much
# slower than NERExact despite better Precision/IoU. Set to True to restore
# the previous behaviour (e.g. for an A/B comparison) without deleting the
# merge implementation itself.
CACD_ENABLE_MERGE = False

MERGE_SAME_DOC_ONLY       = True
MERGE_MAX_SIZE_MULTIPLIER = 2.0
MERGE_MAX_EMBED_CHARS     = 900


# NIS_SENTENCE_NOVEL: min NIS for a sentence in A to be considered novel.
#   Uses MIN rule across K candidates: sᵢ must be novel relative to ALL Bⱼ.
#   Value 0.7 sits between partial overlap (~0.6) and fully novel (~0.9).
NIS_SENTENCE_NOVEL = 0.55

# MIN_NOVEL_CHARS: minimum character length for the merged novel text to be
#   worth indexing. Below this the novel content is too short to produce a
#   meaningful embedding and is discarded instead of being merged.
MIN_NOVEL_CHARS = 50