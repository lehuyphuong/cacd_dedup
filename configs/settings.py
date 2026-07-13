"""
Central configuration for cacd-dedup.

CACD pipeline (3 stages):
  Stage 0: Embedding        -- all-MiniLM-L6-v2, 384-dim
  Stage 1: Coarse retrieval -- exact top-K search over an in-memory pool
                                of already-kept chunk embeddings
  Stage 2: Cross-attention  -- cross-encoder/msmarco-MiniLM-L6-en-de-v1,
                                extract attention matrix, compute NIS
  Stage 3: Decision         -- 3-zone logic using prob_duplicate + NIS
                                + length-aware guard

Evaluation metrics: Precision, Recall, IoU, index size (chunk count +
storage MB).
"""

from pathlib import Path

import torch

# Paths
ROOT_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT_DIR / "data"
RESULTS_DIR  = ROOT_DIR / "results"
QDRANT_PATH  = DATA_DIR / "qdrant_storage"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
QDRANT_PATH.mkdir(exist_ok=True)

# QDRANT_URL: if set (e.g. "http://localhost:6333"), vector_store.get_client()
# connects to a real Qdrant server instead of the embedded/local client.
# Leave as None to use the embedded client (no server required); Qdrant is
# only used to store the final kept chunks for retrieval evaluation, not
# during the CACD decision loop, so either mode works.
QDRANT_URL = None

# Dataset
DATASET_NAME       = "rajpurkar/squad"
DATASET_SPLIT      = "validation"
MAX_DOCUMENTS:      int | None = None
MAX_EVAL_QUESTIONS: int | None = None

# Embedding -- all-MiniLM-L6-v2 (Stage 0)
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DIM   = 384
EMBED_BATCH_SIZE = 512 if torch.cuda.is_available() else 128

# CACD -- Stage 1 (coarse retrieval)
# Number of chunks processed together per micro-batch during ingestion.
# Chunks within the same micro-batch are all compared against the index as
# it stood at the start of the batch, so they cannot see each other; only
# chunks from earlier batches are visible. Smaller values reduce this
# staleness (1 = fully sequential, no staleness); larger values trade a
# bigger staleness window for fewer, larger cross-encoder forward passes.
CACD_INGEST_BATCH_SIZE = 32

# Number of nearest chunks retrieved from the in-memory index per new chunk.
CACD_TOP_K_CANDIDATES = 5

# Mixed-precision (FP16) inference for the cross-encoder. Only takes effect
# on CUDA; a no-op on CPU-only runs regardless of this value.
CACD_USE_FP16 = True

# CACD -- Stage 2 (cross-attention)
# Pretrained, no fine-tuning. Selected via a comparison across 37 candidate
# cross-encoder models (see scripts/experiment_model_comparison.py).
CACD_CROSS_ENCODER_MODEL = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# When True, only the cross-encoder's final transformer layer computes
# attention weights (the rest use the faster default attention path).
# Novel Information Score only ever reads the final layer's attention, so
# this is functionally identical to requesting attention from every layer,
# just faster. get_cross_encoder() self-tests this at startup and falls
# back automatically to computing attention on every layer if the test
# fails, so this flag is safe to leave on; use it as a manual kill switch
# if needed.
CACD_USE_LAST_LAYER_EAGER_ATTENTION = True

# CACD -- Stage 3 (decision thresholds)
# The KEEP/DROP probability cutoff is derived from a cost ratio rather than
# hand-picked: cutoff = cost_FP / (cost_FP + cost_FN), where cost_FP is the
# cost of wrongly dropping a novel chunk and cost_FN the cost of wrongly
# keeping a duplicate. Symmetric costs (both 1.0) give cutoff = 0.5.
CACD_COST_FALSE_POSITIVE = 1.0
CACD_COST_FALSE_NEGATIVE = 1.0

# Vector store -- Qdrant
COLLECTION_PREFIX = "cacd_dedup"

# Retrieval (final evaluation)
TOP_K = 5

# LLM (Ollama -- optional, not used in current scope)
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "mistral"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 256

SYSTEM_PROMPT = """\
You are a precise question-answering assistant. Answer the question using \
ONLY the provided context passages. Be concise — answer in 1-3 sentences. \
If the answer is not in the context, respond with exactly: "I don't know."\
"""

# Chunking configurations
# 9 strategies x 2 size configurations = 18 total configs.
#   chunk_size : target chunk size in characters (base unit)
#   overlap    : character overlap between chunks (0 for strategies that
#                do not use sliding windows)
#   extra      : strategy-specific overrides (optional)

def _make_configs() -> list[dict]:
    chunker_configs = [
        {"strategy": "FixedSize",   "chunk_size": 200, "overlap": 0},
        {"strategy": "FixedSize",   "chunk_size": 400, "overlap": 0},

        {"strategy": "Recursive",   "chunk_size": 200, "overlap": 0},
        {"strategy": "Recursive",   "chunk_size": 400, "overlap": 0},

        {"strategy": "Semantic",    "chunk_size": 200, "overlap": 0,
         "extra": {"threshold_percentile": 95.0}},
        {"strategy": "Semantic",    "chunk_size": 400, "overlap": 0,
         "extra": {"threshold_percentile": 95.0}},

        {"strategy": "Overlapping", "chunk_size": 400, "overlap": 200},
        {"strategy": "Overlapping", "chunk_size": 800, "overlap": 400},

        {"strategy": "AdaptiveEntropy",       "chunk_size": 300, "overlap": 0},
        {"strategy": "AdaptiveEntropy",       "chunk_size": 500, "overlap": 0},

        {"strategy": "AdaptiveSentenceLen",   "chunk_size": 4,   "overlap": 0,
         "extra": {"target_sentences": 4, "min_sentences": 2, "max_sentences": 8}},
        {"strategy": "AdaptiveSentenceLen",   "chunk_size": 6,   "overlap": 0,
         "extra": {"target_sentences": 6, "min_sentences": 3, "max_sentences": 12}},

        {"strategy": "HierarchicalParentChild", "chunk_size": 200, "overlap": 0,
         "extra": {"parent_size": 600}},
        {"strategy": "HierarchicalParentChild", "chunk_size": 400, "overlap": 0,
         "extra": {"parent_size": 800}},

        {"strategy": "Contextual",            "chunk_size": 300, "overlap": 0},
        {"strategy": "Contextual",            "chunk_size": 500, "overlap": 0},

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
